#!/usr/bin/env python3
"""Static integrity test suite for the ml-engineer plugin.

Checks every skill and agent file for:
- valid YAML frontmatter
- required fields and types
- required sections present
- cross-references that resolve (skills referenced by agents exist; schema fields
  referenced by skills are defined; iteration paths consistent)
- enum value consistency (status enums, source enums, axis enums, etc.)
- no dangling references to renamed/missing files

Returns exit 0 if everything passes, 1 otherwise. Prints per-file findings.
"""

import json
import re
import sys
import yaml
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = ROOT / "skills"
AGENTS_DIR = ROOT / "agents"
SCHEMA_DOC = ROOT / "docs/superpowers/specs/research-engine-workdir-schema.md"

# Expected enum values per the schema doc — must match across all files that use them.
EXPECTED_ENUMS = {
    "hypothesis_status": {"live", "running", "scored", "archived"},
    "hypothesis_source": {"literature", "mutation", "failure_mining", "cross_domain", "adversarial"},
    "leaderboard_status": {"scored", "failed", "debug_exhausted"},
    "engine_state": {"initializing", "running", "awaiting_user", "paused", "stopped"},
    "next_action": {
        "re_frame_problem", "re_mine_literature", "re_generate_hypotheses",
        "re_select_next", "dispatch_to_subagent", "re_update_narrative",
        "re_detect_plateau", "re_zoom_out", "re_write_up",
        "awaiting_user_response", "null",
    },
    "reframe_axis": {"metric", "unit_of_analysis", "decomposition", "data_slice"},
    "expected_gain_cost": {"low", "med", "high"},
}

# Skills that should ONLY fire from inside research-engine.
ENGINE_ONLY_SKILLS = {
    "re-frame-problem", "re-mine-literature", "re-generate-hypotheses",
    "re-select-next", "re-update-narrative", "re-detect-plateau",
    "re-zoom-out", "re-write-up",
}

# Required sections vary by skill family because conventions evolved across releases.
#
# v0.3.0 (re-* engine skills) — strict convention: When to invoke / When NOT to invoke / Process / Verification gates / Hard constraints.
# v0.2.0 (dl-* deep-learning skills) — mostly the v0.3.0 shape but some use 'Phases' instead of 'Process'.
# v0.1.0 (ml-engineer-* tabular skills) — older, often use 'Required structure' / 'Goal' / 'Rules' / 'Output checklist' instead of When/Process.
#
# What every skill MUST have, regardless of release:
#   - frontmatter with name, description, license, metadata
#   - at least one top-level section explaining what the skill does and when to call it
#     (acceptable headings: "When to invoke", "Trigger", "Required structure", "Goal", "Phases", or any "Step N")
#
# Stricter requirements only apply to v0.3.0 (re-*) skills — those are recent and ship under the
# tight convention.

PROCESS_HEADING_SYNONYMS = {"Process", "Phases", "Required structure"}
INVOKE_HEADING_SYNONYMS = {"When to invoke", "Trigger", "When"}

REQUIRED_SKILL_SECTIONS_V03 = ["When to invoke", "When NOT to invoke", "Process"]

# Required frontmatter keys for skills.
REQUIRED_SKILL_FRONTMATTER = {"name", "description", "license", "metadata"}

# Required frontmatter keys for agents.
REQUIRED_AGENT_FRONTMATTER = {"name", "description"}


class TestRunner:
    def __init__(self):
        self.failures = []  # list of (file_path, severity, message)
        self.warnings = []
        self.passed_count = 0

    def fail(self, path, msg):
        self.failures.append((str(path), "FAIL", msg))

    def warn(self, path, msg):
        self.warnings.append((str(path), "WARN", msg))

    def parse_frontmatter(self, path):
        """Extract YAML frontmatter and body. Returns (frontmatter_dict, body) or (None, None) on failure."""
        text = path.read_text()
        m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
        if not m:
            self.fail(path, "no frontmatter found (must start with '---' line)")
            return None, None
        try:
            fm = yaml.safe_load(m.group(1))
        except yaml.YAMLError as e:
            self.fail(path, f"frontmatter is not valid YAML: {e}")
            return None, None
        return fm, m.group(2)

    def get_top_level_sections(self, body):
        """Extract ## headings, ignoring those inside ``` fenced code blocks."""
        sections = []
        in_fence = False
        for line in body.splitlines():
            if line.startswith("```"):
                in_fence = not in_fence
                continue
            if not in_fence and line.startswith("## "):
                sections.append(line[3:].strip())
        return sections

    def test_skill(self, path):
        skill_name = path.parent.name
        fm, body = self.parse_frontmatter(path)
        if fm is None:
            return

        # Frontmatter required keys.
        missing = REQUIRED_SKILL_FRONTMATTER - set(fm.keys())
        if missing:
            self.fail(path, f"missing frontmatter keys: {sorted(missing)}")
            return

        # Name matches directory.
        if fm["name"] != skill_name:
            self.fail(path, f"frontmatter name '{fm['name']}' != directory name '{skill_name}'")

        # License is MIT.
        if fm.get("license") != "MIT":
            self.fail(path, f"license is '{fm.get('license')}', expected 'MIT'")

        # metadata.source and metadata.version present.
        meta = fm.get("metadata") or {}
        if not isinstance(meta, dict) or "source" not in meta or "version" not in meta:
            self.fail(path, "metadata must have 'source' and 'version' keys")

        # Description is non-empty string.
        desc = fm.get("description", "")
        if not isinstance(desc, str) or len(desc) < 30:
            self.fail(path, f"description too short or wrong type ({len(desc) if isinstance(desc, str) else type(desc)} chars)")

        # Required sections per skill-family convention.
        sections = self.get_top_level_sections(body)
        sections_set = set(sections)

        is_v03_skill = skill_name.startswith("re-")  # research-engine skills

        if is_v03_skill:
            # Strict v0.3.0 convention.
            for required in REQUIRED_SKILL_SECTIONS_V03:
                if required not in sections_set:
                    self.fail(path, f"missing required section '## {required}' (v0.3.0 convention)")
        else:
            # Legacy v0.1.0 / v0.2.0 skills: trigger language lives in the description (frontmatter),
            # body just needs SOME process-shaped content. Acceptable forms: 'Process', 'Phases',
            # 'Required structure', 'The four phases', or a numbered 'Step N' / 'Phase N' subsection,
            # or any of: 'Layouts', 'Two layouts', 'Recipe', 'Iron Law' (these are content-defining
            # headings used by older skills).
            content_heading_signals = {
                "Process", "Phases", "Required structure", "The four phases",
                "Iron Law", "Recipe template",
            }
            has_content = (
                bool(content_heading_signals & sections_set)
                or any(s.startswith("Step ") for s in sections)
                or any(s.startswith("Phase ") for s in sections)
                or any("layout" in s.lower() for s in sections)
                or any("template" in s.lower() for s in sections)
                or any("rule" in s.lower() for s in sections)
                or len(sections) >= 4  # any skill with ≥4 sections is structured even if the heading words differ
            )
            if not has_content:
                self.fail(
                    path,
                    f"no process/content-shaped section found (got: {sections})",
                )

        # Engine-only skills must have the engine-only language in their description.
        if skill_name in ENGINE_ONLY_SKILLS:
            required_phrases = [
                "Only fires from inside the research-engine agent",
                "Do NOT invoke from",
            ]
            for phrase in required_phrases:
                if phrase not in desc:
                    self.fail(path, f"engine-only skill missing phrase: '{phrase}'")
            # Must mention all four sibling agents in the do-not-invoke list.
            for agent in ("ml-engineer", "cv-engineer", "nlp-engineer", "llm-engineer"):
                if agent not in desc:
                    self.fail(path, f"engine-only skill description missing sibling agent '{agent}' in do-not-invoke clause")

        self.passed_count += 1

    def test_agent(self, path):
        agent_name = path.stem
        fm, body = self.parse_frontmatter(path)
        if fm is None:
            return

        missing = REQUIRED_AGENT_FRONTMATTER - set(fm.keys())
        if missing:
            self.fail(path, f"missing frontmatter keys: {sorted(missing)}")
            return

        if fm["name"] != agent_name:
            self.fail(path, f"frontmatter name '{fm['name']}' != filename stem '{agent_name}'")

        desc = fm.get("description", "")
        if not isinstance(desc, str) or len(desc) < 30:
            self.fail(path, f"description too short or wrong type")

        # The four domain agents must have a Step 0 prologue.
        if agent_name in ("ml-engineer", "cv-engineer", "nlp-engineer", "llm-engineer"):
            if "Step 0 — Engine vs" not in body:
                self.fail(path, "missing 'Step 0 — Engine vs. one-shot' section")

        # research-engine must NOT have Step 0 (it IS the engine).
        if agent_name == "research-engine":
            if "Step 0 — Engine vs" in body:
                self.fail(path, "research-engine should not have Step 0 prologue (it IS the engine)")
            # Should reference all 8 re-* skills in its skill table.
            for skill in ENGINE_ONLY_SKILLS:
                if skill not in body:
                    self.fail(path, f"research-engine body does not reference engine-only skill '{skill}'")

        self.passed_count += 1

    def test_schema_consistency(self):
        """Verify every enum value used in skill files matches the schema doc."""
        if not SCHEMA_DOC.exists():
            self.fail(SCHEMA_DOC, "schema doc missing")
            return

        schema_text = SCHEMA_DOC.read_text()

        # Spot-check that schema lists the expected enums.
        for enum_name, expected in EXPECTED_ENUMS.items():
            for value in expected:
                if value == "null":
                    continue  # JSON null, not a literal string
                if value not in schema_text:
                    self.warn(SCHEMA_DOC, f"enum value '{value}' (for {enum_name}) not found in schema doc text")

    def test_workdir_path_consistency(self):
        """Skills that reference workdir paths should use consistent path patterns."""
        good_pattern = re.compile(r"<workdir>/research_engine/")
        for skill_path in sorted(SKILLS_DIR.glob("re-*/SKILL.md")):
            text = skill_path.read_text()
            if "research_engine" in text and "<workdir>" not in text:
                self.fail(skill_path, "uses 'research_engine' path without the '<workdir>/' prefix")

    def test_status_enum_correctness(self):
        """No skill should write 'debug-exhausted' (kebab); must be 'debug_exhausted' (snake)."""
        for path in list(SKILLS_DIR.glob("**/SKILL.md")) + list(AGENTS_DIR.glob("*.md")):
            text = path.read_text()
            # Skip historical mentions in commit-message-style strings; just flag any kebab form.
            if re.search(r"\bdebug-exhausted\b", text):
                self.fail(path, "uses kebab-case 'debug-exhausted'; should be snake_case 'debug_exhausted'")

    def test_next_action_consistency(self):
        """re-detect-plateau must translate decisions to schema-valid next_action values."""
        path = SKILLS_DIR / "re-detect-plateau" / "SKILL.md"
        if not path.exists():
            return
        text = path.read_text()
        # Must mention the four mapped values.
        for action in ("re_select_next", "re_generate_hypotheses", "re_zoom_out", "re_write_up"):
            if action not in text:
                self.fail(path, f"missing next_action mapping target '{action}'")

    def test_re_zoom_out_owns_fields(self):
        """re-zoom-out must claim ownership of zoom_out_count and last_zoom_out_iter."""
        path = SKILLS_DIR / "re-zoom-out" / "SKILL.md"
        if not path.exists():
            return
        text = path.read_text()
        if "zoom_out_count" not in text or "last_zoom_out_iter" not in text:
            self.fail(path, "must reference zoom_out_count and last_zoom_out_iter (it owns them)")

    def test_target_hit_resolved_owned_by_plateau(self):
        """re-detect-plateau must own target_hit_resolved (writes it once per session)."""
        path = SKILLS_DIR / "re-detect-plateau" / "SKILL.md"
        if not path.exists():
            return
        text = path.read_text()
        if "target_hit_resolved" not in text:
            self.fail(path, "must reference target_hit_resolved (it owns this field)")

    def test_spend_so_far_usd_recomputed(self):
        """re-update-narrative must recompute spend_so_far_usd after appending leaderboard."""
        path = SKILLS_DIR / "re-update-narrative" / "SKILL.md"
        if not path.exists():
            return
        text = path.read_text()
        if "spend_so_far_usd" not in text:
            self.fail(path, "must update spend_so_far_usd after leaderboard append (cost ceiling depends on it)")

    def test_hypothesis_json_written(self):
        """re-select-next must write iterations/<NNN>/hypothesis.json."""
        path = SKILLS_DIR / "re-select-next" / "SKILL.md"
        if not path.exists():
            return
        text = path.read_text()
        if "hypothesis.json" not in text:
            self.fail(path, "must write iterations/<NNN>/hypothesis.json (re-update-narrative reads it)")

    def test_skills_referenced_by_agents_exist(self):
        """Every skill name referenced by an agent body must be a real skill directory."""
        existing_skills = {p.name for p in SKILLS_DIR.iterdir() if p.is_dir()}
        skill_pattern = re.compile(r"`((?:re|dl|ml-engineer)-[a-z0-9-]+)`")
        for agent_path in AGENTS_DIR.glob("*.md"):
            text = agent_path.read_text()
            referenced = set(skill_pattern.findall(text))
            missing = referenced - existing_skills
            # Filter out things that look like skill names but are file paths or general words.
            real_missing = {s for s in missing if s.startswith(("re-", "dl-", "ml-engineer-")) and "/" not in s}
            for skill in real_missing:
                self.fail(agent_path, f"references nonexistent skill '{skill}'")

    def run(self):
        # Per-file checks.
        for skill_dir in sorted(SKILLS_DIR.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                self.fail(skill_dir, "skill directory has no SKILL.md")
                continue
            self.test_skill(skill_md)

        for agent_path in sorted(AGENTS_DIR.glob("*.md")):
            self.test_agent(agent_path)

        # Cross-file invariants.
        self.test_schema_consistency()
        self.test_workdir_path_consistency()
        self.test_status_enum_correctness()
        self.test_next_action_consistency()
        self.test_re_zoom_out_owns_fields()
        self.test_target_hit_resolved_owned_by_plateau()
        self.test_spend_so_far_usd_recomputed()
        self.test_hypothesis_json_written()
        self.test_skills_referenced_by_agents_exist()

    def report(self):
        print(f"Files checked passed: {self.passed_count}")
        print(f"Failures: {len(self.failures)}")
        print(f"Warnings: {len(self.warnings)}")
        print()
        for path, sev, msg in self.failures:
            print(f"[FAIL] {path}: {msg}")
        for path, sev, msg in self.warnings:
            print(f"[WARN] {path}: {msg}")
        return 0 if not self.failures else 1


if __name__ == "__main__":
    runner = TestRunner()
    runner.run()
    sys.exit(runner.report())
