from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    path: Path
    required_toolsets: set[str] = field(default_factory=set)
    allowed_context: set[str] = field(default_factory=set)

    def metadata(self, enabled_toolsets: set[str]) -> dict[str, object]:
        missing = sorted(self.required_toolsets - enabled_toolsets)
        return {
            "name": self.name,
            "description": self.description,
            "path": str(self.path),
            "required_toolsets": sorted(self.required_toolsets),
            "allowed_context": sorted(self.allowed_context),
            "available": not missing,
            "missing_toolsets": missing,
        }


class SkillManager:
    def __init__(self, skills_dir: str | Path):
        self.skills_dir = Path(skills_dir)

    def discover(self, enabled_toolsets: set[str]) -> list[Skill]:
        if not self.skills_dir.exists():
            return []
        skills: list[Skill] = []
        for skill_md in sorted(self.skills_dir.glob("*/SKILL.md")):
            skill = self._parse(skill_md)
            if skill.required_toolsets <= enabled_toolsets:
                skills.append(skill)
        return skills

    def index_prompt(self, enabled_toolsets: set[str]) -> str:
        skills = self.discover(enabled_toolsets)
        if not skills:
            return ""
        lines = ["<available_skills>"]
        for skill in skills:
            lines.append(f"- {skill.name}: {skill.description}")
        lines.append("</available_skills>")
        lines.append("Use the skill_view tool to load full skill instructions only when needed.")
        return "\n".join(lines)

    def view(self, name: str, enabled_toolsets: set[str]) -> dict[str, object]:
        for skill in self.discover(enabled_toolsets):
            if skill.name == name:
                return {
                    "metadata": skill.metadata(enabled_toolsets),
                    "content": skill.path.read_text(encoding="utf-8"),
                }
        raise FileNotFoundError(f"skill not found or unavailable: {name}")

    @staticmethod
    def _parse(path: Path) -> Skill:
        text = path.read_text(encoding="utf-8")
        name = path.parent.name
        description = ""
        required_toolsets: set[str] = set()
        allowed_context: set[str] = set()
        for raw in text.splitlines():
            line = raw.strip()
            if line.startswith("# "):
                name = line[2:].strip() or name
            elif line.lower().startswith("description:"):
                description = line.split(":", 1)[1].strip()
            elif line.lower().startswith("required_toolsets:"):
                values = line.split(":", 1)[1].strip()
                required_toolsets = {item.strip() for item in values.split(",") if item.strip()}
            elif line.lower().startswith("allowed_context:"):
                values = line.split(":", 1)[1].strip()
                allowed_context = {item.strip() for item in values.split(",") if item.strip()}
            if name and description:
                continue
        if not description:
            description = _first_non_heading_line(text) or "No description provided."
        return Skill(
            name=name,
            description=description,
            path=path,
            required_toolsets=required_toolsets,
            allowed_context=allowed_context,
        )


def _first_non_heading_line(text: str) -> str:
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            return line[:200]
    return ""
