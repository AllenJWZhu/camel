# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml

from camel.agents.subagents.spec import SubAgentSpec
from camel.logger import get_logger

logger = get_logger(__name__)

_VALID_SUB_AGENT_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class SubAgentRegistry:
    r"""Loads named sub-agent specifications from markdown files."""

    def __init__(
        self,
        search_paths: Optional[List[Union[str, Path]]] = None,
        working_directory: Optional[Path] = None,
    ) -> None:
        self._working_directory = Path(working_directory or Path.cwd())
        self._search_paths = search_paths

    def list_specs(self) -> Dict[str, SubAgentSpec]:
        r"""Load all available specs in priority order.

        Priority order:
            1. repo: ``.camel/agents`` under current working directory
            2. user: ``~/.camel/agents``

        If ``search_paths`` was provided, those paths are used in the given
        order and tagged as ``custom`` scope.
        """
        specs: Dict[str, SubAgentSpec] = {}
        for scope, root in self._iter_roots():
            if not root.is_dir():
                continue
            for file_path in sorted(root.rglob("*.md")):
                if self._is_hidden_path(file_path, root):
                    continue
                spec = self._parse_spec_file(file_path, scope)
                if spec is None:
                    continue
                if spec.name in specs:
                    continue
                specs[spec.name] = spec
        return specs

    def _iter_roots(self) -> List[Tuple[str, Path]]:
        if self._search_paths is not None:
            return [("custom", Path(path).expanduser()) for path in self._search_paths]
        return [
            ("repo", self._working_directory / ".camel" / "agents"),
            ("user", Path.home() / ".camel" / "agents"),
        ]

    def _parse_spec_file(
        self, file_path: Path, scope: str
    ) -> Optional[SubAgentSpec]:
        try:
            contents = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read sub-agent spec %s: %s", file_path, exc)
            return None

        frontmatter_text, body = self._split_frontmatter(contents)
        if frontmatter_text is None:
            logger.warning(
                "Skipping sub-agent spec without YAML frontmatter: %s",
                file_path,
            )
            return None

        try:
            data = yaml.safe_load(frontmatter_text) or {}
        except yaml.YAMLError as exc:
            logger.warning(
                "Invalid YAML frontmatter in sub-agent spec %s: %s",
                file_path,
                exc,
            )
            return None

        if not isinstance(data, dict):
            logger.warning(
                "Sub-agent frontmatter must be a mapping in %s", file_path
            )
            return None

        raw_name = data.get("name")
        raw_description = data.get("description")
        if not isinstance(raw_name, str) or not isinstance(raw_description, str):
            logger.warning(
                "Sub-agent spec missing required 'name'/'description' in %s",
                file_path,
            )
            return None

        name = raw_name.strip()
        description = raw_description.strip()
        if not name or not description:
            logger.warning(
                "Sub-agent spec has empty 'name' or 'description' in %s",
                file_path,
            )
            return None

        if _VALID_SUB_AGENT_NAME_PATTERN.fullmatch(name) is None:
            logger.warning(
                "Skipping sub-agent with invalid name '%s' in %s",
                name,
                file_path,
            )
            return None

        system_prompt = body.strip()
        if not system_prompt:
            logger.warning(
                "Sub-agent spec '%s' has empty body prompt in %s",
                name,
                file_path,
            )
            return None

        allowed_tools = self._normalize_tools(data.get("tools"), file_path)
        if allowed_tools == []:
            logger.warning(
                "Sub-agent spec '%s' declares an empty tools list in %s",
                name,
                file_path,
            )

        return SubAgentSpec(
            name=name,
            description=description,
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
            path=file_path,
            scope=scope,
        )

    def _normalize_tools(
        self, raw_tools: object, file_path: Path
    ) -> Optional[list[str]]:
        if raw_tools is None:
            return None

        if isinstance(raw_tools, str):
            candidates = [part.strip() for part in raw_tools.split(",")]
        elif isinstance(raw_tools, list):
            candidates = []
            for item in raw_tools:
                if not isinstance(item, str):
                    logger.warning(
                        "Sub-agent tools must be strings in %s; got %r",
                        file_path,
                        item,
                    )
                    return None
                candidates.append(item.strip())
        else:
            logger.warning(
                "Sub-agent tools field must be string or list in %s",
                file_path,
            )
            return None

        normalized: list[str] = []
        for candidate in candidates:
            if not candidate:
                continue
            if candidate not in normalized:
                normalized.append(candidate)
        return normalized

    def _split_frontmatter(self, content: str) -> Tuple[Optional[str], str]:
        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            return None, content

        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                frontmatter = "\n".join(lines[1:index])
                body = "\n".join(lines[index + 1 :])
                return frontmatter, body
        return None, content

    def _is_hidden_path(self, path: Path, root: Path) -> bool:
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = path
        return any(part.startswith(".") for part in relative.parts)

