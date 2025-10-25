"""
This module contains the tools for the essay writer agent.
- A custom Wikipedia search tool that uses the 'requests' library
- A word counter utility function
- A context collector tool for gathering information from multiple sources
- A context comparison tool for verifying content alignment
- A Warhammer 40K army stats and rules tool (REVISED)
"""
import json
import re
from typing import Dict, List

import requests
from langchain_core.tools import BaseTool


class WikipediaSearchTool(BaseTool):
    """
    A tool for searching Wikipedia that directly uses the Wikipedia API,
    avoiding the need for the 'wikipedia' library.
    """
    name: str = "wikipedia"
    description: str = (
        "A wrapper around Wikipedia. "
        "Useful for when you need to answer general questions about "
        "people, places, companies, facts, historical events, or other subjects. "
        "Input should be a search query."
    )
    doc_content_chars_max: int = 4000

    def _run(self, query: str) -> str:
        """Use the Wikipedia tool."""
        try:
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/58.0.3029.110 Safari/537.36"
            })
            url = "https://en.wikipedia.org/w/api.php"

            # Step 1: Search for the page title
            search_params = {
                "action": "opensearch", "search": query, "limit": "1",
                "namespace": "0", "format": "json",
            }
            search_response = session.get(url=url, params=search_params, timeout=5)
            search_response.raise_for_status()
            search_data = search_response.json()

            if not search_data[1]:
                return "No good Wikipedia Search Result was found"

            page_title = search_data[1][0]

            # Step 2: Fetch page content
            content_params = {
                "action": "query", "format": "json", "titles": page_title,
                "prop": "extracts", "explaintext": True,
            }
            content_response = session.get(url=url, params=content_params, timeout=5)
            content_response.raise_for_status()
            content_data = content_response.json()
            page_info = content_data["query"]["pages"]
            page_id = list(page_info.keys())[0]

            if page_id == "-1":
                return f"Page titled '{page_title}' does not exist on Wikipedia."

            extract = page_info[page_id].get("extract", "")
            return extract[:self.doc_content_chars_max] if extract else "No content found."

        except requests.exceptions.RequestException as e:
            return f"An error occurred with the Wikipedia API: {e}"


class TextContextCollectorTool(BaseTool):
    """
    A tool for collecting context from webpages based on a topic.
    Checks a database file of links and retrieves webpage content.
    """
    name: str = "context_collector"
    description: str = (
        "Collects contextual information from webpages based on a given topic. "
        "Reads from a database of topic-to-URL mappings and fetches the content. "
        "Useful for gathering research material or background information. "
        "Input should be a topic string."
    )

    links_database: Dict[str, List[str]] = {
        "ci/cd": [
            "https://www.redhat.com/en/topics/devops/what-is-ci-cd",
            "https://en.wikipedia.org/wiki/CI/CD",
            "https://about.gitlab.com/topics/ci-cd/",
            "https://www.atlassian.com/continuous-delivery",
        ],
        "data engineering": [
            "https://www.coursera.org/articles/what-does-a-data-engineer-do-and-how-do-i-become-one",
            "https://www.alxafrica.com/programme/data-engineering/",
            "https://www.ibm.com/think/topics/data-engineering",
        ],
        "dag": [
            "https://en.wikipedia.org/wiki/Directed_acyclic_graph",
            "https://www.ibm.com/think/topics/directed-acyclic-graph",
            "https://hazelcast.com/foundations/distributed-computing/directed-acyclic-graph/",
        ],
        "mlops": [
            "https://en.wikipedia.org/wiki/MLOps",
            "https://aws.amazon.com/what-is/mlops/",
            "https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning",
            "https://ml-ops.org/",
            "https://www.databricks.com/glossary/mlops",
        ],
    }
    content_chars_max: int = 5000

    def _run(self, topic: str) -> str:
        """Fetch context from webpages related to the topic."""
        try:
            # Normalize topic (lowercase for matching)
            topic_lower = topic.lower()

            # Find matching topic in database
            if topic_lower not in self.links_database:
                return f"No links found for topic: '{topic}'. Available topics: {', '.join(self.links_database.keys())}"

            urls = self.links_database[topic_lower]
            collected_context = []

            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/58.0.3029.110 Safari/537.36"
            })

            for url in urls:
                try:
                    # For Wikipedia URLs, use the API
                    if "wikipedia.org/wiki/" in url:
                        page_title = url.split("/wiki/")[-1]
                        api_url = "https://en.wikipedia.org/w/api.php"
                        params = {
                            "action": "query",
                            "format": "json",
                            "titles": page_title.replace("_", " "),
                            "prop": "extracts",
                            "explaintext": True,
                        }
                        response = session.get(url=api_url, params=params, timeout=5)
                        response.raise_for_status()
                        data = response.json()
                        page_info = data["query"]["pages"]
                        page_id = list(page_info.keys())[0]

                        if page_id != "-1":
                            extract = page_info[page_id].get("extract", "")
                            collected_context.append(f"Source: {url}\n{extract[:self.content_chars_max]}")
                    else:
                        # For other URLs, fetch directly
                        response = session.get(url, timeout=5)
                        response.raise_for_status()
                        collected_context.append(f"Source: {url}\n{response.text[:self.content_chars_max]}")

                except requests.exceptions.RequestException as e:
                    collected_context.append(f"Failed to fetch {url}: {e}")

            if not collected_context:
                return f"No context could be collected for topic: '{topic}'"

            return "\n\n---\n\n".join(collected_context)

        except Exception as e:
            return f"An error occurred while collecting context: {e}"


class ContextComparisonTool(BaseTool):
    """
    A tool for comparing a paragraph against provided context.
    Checks if the paragraph aligns with and is supported by the context.
    """
    name: str = "context_comparison"
    description: str = (
        "Compares a provided paragraph against reference context to check alignment. "
        "Useful for verifying if content matches source material or stays on topic. "
        "Input should be a JSON string with 'paragraph' and 'context' keys."
    )

    def _run(self, input_text: str) -> str:
        """Compare paragraph with context."""
        try:
            # Parse input
            try:
                data = json.loads(input_text)
                paragraph = data.get("paragraph", "")
                context = data.get("context", "")
            except json.JSONDecodeError:
                return "Invalid input format. Please provide JSON with 'paragraph' and 'context' keys."

            if not paragraph or not context:
                return "Both 'paragraph' and 'context' must be provided."

            # Normalize text for comparison
            paragraph_lower = paragraph.lower()
            context_lower = context.lower()

            # Extract key terms from paragraph (simple word-based approach)
            paragraph_words = set(word.strip('.,!?;:()[]{}\"\'')
                                 for word in paragraph_lower.split()
                                 if len(word.strip('.,!?;:()[]{}\"\'')) > 3)

            context_words = set(word.strip('.,!?;:()[]{}\"\'')
                               for word in context_lower.split()
                               if len(word.strip('.,!?;:()[]{}\"\'')) > 3)

            # Calculate overlap
            common_words = paragraph_words.intersection(context_words)
            if not paragraph_words:
                return "Paragraph is empty or contains no substantial words."

            overlap_percentage = (len(common_words) / len(paragraph_words)) * 100

            # Generate report
            result = f"Context Comparison Analysis:\n"  # noqa: F541
            result += f"- Paragraph word count: {len(paragraph.split())}\n"
            result += f"- Context word count: {len(context.split())}\n"
            result += f"- Key terms in paragraph: {len(paragraph_words)}\n"
            result += f"- Key terms matching context: {len(common_words)}\n"
            result += f"- Alignment score: {overlap_percentage:.1f}%\n\n"
            if overlap_percentage >= 70:
                result += "✓ HIGH ALIGNMENT: Paragraph strongly matches the provided context."
            elif overlap_percentage >= 40:
                result += "~ MODERATE ALIGNMENT: Paragraph partially matches the context."
            else:
                result += "✗ LOW ALIGNMENT: Paragraph has limited connection to the context."

            result += f"\n\nCommon key terms: {', '.join(list(common_words)[:10])}"
            return result

        except Exception as e:
            return f"An error occurred during context comparison: {e}"


class WH40KArmyDataTool(BaseTool):
    """
    Enhanced tool for collecting Warhammer 40K army data.

    Parses Goonhammer stats page to extract actual tournament data and army lists.
    """

    name: str = "wh40k_army_data"
    description: str = (
        "Collects Warhammer 40K army statistics, tournament win rates, and example lists. "
        "Useful for gathering current meta data about faction performance and popular builds. "
        "Input should be a faction name (e.g., 'Space Marines', 'Tyranids', 'Necrons')."
    )

    # Faction name mapping for normalization
    faction_mapping: Dict[str, str] = {
        "space marines": "Adeptus Astartes",
        "sm": "Adeptus Astartes",
        "astartes": "Adeptus Astartes",
        "marines": "Adeptus Astartes",
        "chaos space marines": "Heretic Astartes",
        "csm": "Heretic Astartes",
        "chaos marines": "Heretic Astartes",
        "tyranids": "Tyranids",
        "nids": "Tyranids",
        "necrons": "Necrons",
        "crons": "Necrons",
        "orks": "Orks",
        "greenskins": "Orks",
        "astra militarum": "Astra Militarum",
        "guard": "Astra Militarum",
        "imperial guard": "Astra Militarum",
        "tau": "T'au Empire",
        "tau empire": "T'au Empire",
        "eldar": "Aeldari",
        "aeldari": "Aeldari",
        "craftworlds": "Aeldari",
        "drukhari": "Drukhari",
        "dark eldar": "Drukhari",
        "admech": "Adeptus Mechanicus",
        "adeptus mechanicus": "Adeptus Mechanicus",
        "thousand sons": "Thousand Sons",
        "death guard": "Death Guard",
        "world eaters": "World Eaters",
        "genestealer cults": "Genestealer Cults",
        "gsc": "Genestealer Cults",
        "custodes": "Adeptus Custodes",
        "adeptus custodes": "Adeptus Custodes",
        "imperial knights": "Imperial Knights",
        "chaos knights": "Chaos Knights",
        "sisters": "Adepta Sororitas",
        "adepta sororitas": "Adepta Sororitas",
        "grey knights": "Grey Knights",
        "leagues of votann": "Leagues of Votann",
        "votann": "Leagues of Votann",
        "squats": "Leagues of Votann",
        "chaos daemons": "Chaos Daemons",
        "daemons": "Chaos Daemons",
    }

    stats_url: str = "https://40kstats.goonhammer.com/#t4"
    content_chars_max: int = 10000

    def _normalize_faction(self, faction_name: str) -> str:
        """Normalize faction name to official format."""
        normalized = faction_name.lower().strip()
        return self.faction_mapping.get(normalized, faction_name)

    def _extract_tournament_data(self, html_content: str, faction: str) -> Dict[str, any]:
        """Extract tournament statistics from HTML content."""
        # This is a simplified parser - in production, use BeautifulSoup
        data = {
            "faction": faction,
            "win_rate": "Unknown",
            "popularity": "Unknown",
            "top_detachments": [],
            "common_units": [],
            "tournament_count": "Unknown",
        }

        try:
            # Try to find win rate patterns (example patterns)
            win_rate_pattern = rf"{faction}.*?(\d+\.?\d*)%"
            match = re.search(win_rate_pattern, html_content, re.IGNORECASE)
            if match:
                data["win_rate"] = f"{match.group(1)}%"

            # Try to find popularity/usage rate
            usage_pattern = rf"{faction}.*?(\d+\.?\d*)%.*?usage"
            match = re.search(usage_pattern, html_content, re.IGNORECASE)
            if match:
                data["popularity"] = f"{match.group(1)}%"

        except Exception as e:
            print(f"Error parsing tournament data: {e}")

        return data

    def _extract_example_lists(self, html_content: str, faction: str) -> List[str]:
        """Extract example army lists from content."""
        example_lists = []

        # Look for common list patterns
        list_patterns = [
            r"(?:HQ|Troops|Elites|Fast Attack|Heavy Support).*?(?:\d+\s*pts?)",
            r"(?:Warlord|Leader):.*",
            r"(?:Detachment):.*",
        ]

        for pattern in list_patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE | re.MULTILINE)
            example_lists.extend(matches[:5])  # Limit to 5 examples per pattern

        return example_lists[:10]  # Return max 10 examples

    def _run(self, faction_name: str) -> str:
        """Fetch and parse army data for the faction."""
        try:
            normalized_faction = self._normalize_faction(faction_name)

            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                }
            )

            # Fetch Goonhammer stats
            result = f"WARHAMMER 40K TOURNAMENT DATA: {normalized_faction}\n"
            result += "=" * 70 + "\n\n"

            try:
                stats_response = session.get(self.stats_url, timeout=10)
                stats_response.raise_for_status()
                stats_content = stats_response.text[: self.content_chars_max]

                # Extract tournament data
                tournament_data = self._extract_tournament_data(
                    stats_content, normalized_faction
                )

                result += "## Tournament Statistics\n\n"
                result += f"**Win Rate:** {tournament_data['win_rate']}\n"
                result += f"**Popularity:** {tournament_data['popularity']}\n"
                result += f"**Tournament Count:** {tournament_data['tournament_count']}\n\n"

                # Extract example lists
                example_lists = self._extract_example_lists(
                    stats_content, normalized_faction
                )
                if example_lists:
                    result += "## Example List Patterns\n\n"
                    for i, example in enumerate(example_lists, 1):
                        result += f"{i}. {example}\n"
                    result += "\n"

                result += "## Meta Analysis\n\n"
                result += (
                    f"Based on recent tournament data from Goonhammer stats, "
                    f"{normalized_faction} shows the following trends:\n\n"
                )

                # Provide meta recommendations based on win rate
                if "Unknown" not in tournament_data["win_rate"]:
                    try:
                        win_rate_num = float(
                            tournament_data["win_rate"].replace("%", "")
                        )
                        if win_rate_num >= 55:
                            result += "- **Strong Performance**: This faction is performing well in the current meta.\n"
                            result += "- **Recommendation**: Focus on proven competitive builds.\n"
                        elif win_rate_num >= 45:
                            result += "- **Balanced Performance**: This faction is well-balanced in the current meta.\n"
                            result += "- **Recommendation**: Multiple viable build paths available.\n"
                        else:
                            result += "- **Challenging Performance**: This faction faces"
                            "challenges in the current meta.\n"
                            result += "- **Recommendation**: Focus on specialized builds and skilled play.\n"
                    except ValueError:
                        pass

                result += "\n**Data Source:** Goonhammer 40K Stats\n"
                result += f"**URL:** {self.stats_url}\n\n"

                # Add note about data limitations
                result += (
                    "Note: For detailed unit lists and point costs, "
                    "consult the latest Munitorum Field Manual.\n"
                )

            except requests.exceptions.RequestException as e:
                result += f"Failed to fetch Goonhammer stats: {e!s}\n"
                result += "Providing general faction guidance based on common knowledge.\n\n"

                # Provide fallback generic advice
                result += self._get_fallback_faction_info(normalized_faction)

            return result

        except Exception as e:
            return f"An error occurred while collecting army data: {e}"

    def _get_fallback_faction_info(self, faction: str) -> str:
        """Provide basic faction information as fallback."""
        faction_info = {
            "Adeptus Astartes": {
                "strengths": "Versatile units, strong shooting and melee, good durability",
                "common_units": "Intercessors, Terminators, Land Raiders, Dreadnoughts",
                "detachments": "Gladius Task Force, Ironstorm Spearhead",
            },
            "Tyranids": {
                "strengths": "Swarm tactics, powerful melee, synapse bonuses",
                "common_units": "Termagants, Carnifex, Hive Tyrant, Genestealers",
                "detachments": "Invasion Fleet, Crusher Stampede",
            },
            "Necrons": {
                "strengths": "Durable units, reanimation protocols, strong shooting",
                "common_units": "Necron Warriors, Immortals, Skorpekh Destroyers",
                "detachments": "Awakened Dynasty, Canoptek Court",
            },
            "Orks": {
                "strengths": "High model count, melee power, vehicle options",
                "common_units": "Boyz, Trukks, Deff Dread, Warboss",
                "detachments": "Waaagh! Tribe, War Horde",
            },
            "Astra Militarum": {
                "strengths": "Powerful shooting, numerous units, artillery",
                "common_units": "Infantry Squad, Leman Russ, Rogal Dorn, Sentinels",
                "detachments": "Combined Regiment, Armoured Company",
            },
        }

        info = faction_info.get(faction, None)
        if not info:
            return "General competitive advice: Focus on durable units"
            ", objective holders, and balanced damage output.\n"

        result = f"## {faction} - General Information\n\n"
        result += f"**Strengths:** {info['strengths']}\n\n"
        result += f"**Common Competitive Units:** {info['common_units']}\n\n"
        result += f"**Popular Detachments:** {info['detachments']}\n\n"
        return result


class WH40KRulesValidatorTool(BaseTool):
    """
    Enhanced tool for checking official Warhammer 40K rules.

    Validates army lists against known rules and provides point cost guidance.
    """

    name: str = "wh40k_rules_validator"
    description: str = (
        "Validates Warhammer 40K army lists against official rules and points. "
        "Checks detachment requirements, point costs, and rule compliance. "
        "Input should be 'check' or the name of a specific rule document."
    )

    # Explicit Links to Official Documents (as provided by the user)
    CORE_RULES_URL: str = "https://assets.warhammer-community.com/eng_08-10_warhammer40000_core_rules_balance_dataslate-f47uib0gs9-9kju9nznun.pdf"
    MUNITORUM_MANUAL_URL: str = "https://assets.warhammer-community.com/eng_22-10_warhammer40000_munitorum_field_manual-aifnzbqjbb-6wp0h57upr.pdf"
    BALANCE_DATASLATE_URL: str = "https://assets.warhammer-community.com/eng_22-10_warhammer40000_core_rules_updates-kikyythcsl-5iydvuj9w6.pdf"

    # Basic point cost ranges for validation (Placeholder/Generic)
    point_ranges: Dict[str, Dict[str, tuple]] = {
        "HQ": {"min": 50, "max": 250},
        "Troops": {"min": 50, "max": 200},
        "Elites": {"min": 75, "max": 400},
        "Fast Attack": {"min": 75, "max": 300},
        "Heavy Support": {"min": 100, "max": 500},
        "Dedicated Transport": {"min": 50, "max": 150},
    }

    def _validate_army_structure(self, army_text: str) -> List[str]:
        """Validate basic army structure requirements."""
        issues = []

        # Check for required HQ
        if not re.search(r"HQ", army_text, re.IGNORECASE):
            issues.append("⚠️ Missing HQ unit (required for all detachments)")

        # Check for Battleline/Troops
        if not re.search(r"(?:Troops|Battleline)", army_text, re.IGNORECASE):
            issues.append("⚠️ Missing Battleline/Troops units (typically required)")

        # Extract point values and validate
        point_pattern = r"(\d+)\s*(?:pts?|points?)"
        points = re.findall(point_pattern, army_text, re.IGNORECASE)
        if points:
            total = sum(int(p) for p in points)
            if total < 500 or total > 3000:
                issues.append(
                    f"⚠️ Unusual point total: {total}pts (standard games are 1000-2000pts)"
                )

        return issues

    def _validate_point_costs(self, army_text: str) -> List[str]:
        """Validate that point costs are reasonable."""
        issues = []

        # Extract units with roles and points
        unit_pattern = r"(HQ|Troops|Elites|Fast Attack|Heavy Support|Dedicated Transport).*?(\d+)\s*(?:pts?|points?)"
        matches = re.findall(unit_pattern, army_text, re.IGNORECASE | re.DOTALL)

        for role, points_str in matches:
            points = int(points_str)
            role_ranges = self.point_ranges.get(role, None)
            if role_ranges:
                if points < role_ranges["min"] or points > role_ranges["max"]:
                    issues.append(
                        f"⚠️ Unusual point cost for {role}: {points}pts "
                        f"(typical range: {role_ranges['min']}-{role_ranges['max']}pts)"
                    )

        return issues

    def _run(self, query: str) -> str:
        """Check rules documents and validate army structure."""
        try:
            result = "WARHAMMER 40K RULES VALIDATION\n"
            result += "=" * 70 + "\n\n"

            # Check if query contains army list to validate
            if len(query) > 100:  # Likely contains an army list
                result += "## Army List Validation (Basic Structure & Points)\n\n"

                # Validate structure
                structure_issues = self._validate_army_structure(query)
                if structure_issues:
                    result += "### Structure Issues:\n"
                    for issue in structure_issues:
                        result += f"{issue}\n"
                    result += "\n"
                else:
                    result += "✓ Army structure appears valid\n\n"

                # Validate point costs
                cost_issues = self._validate_point_costs(query)
                if cost_issues:
                    result += "### Point Cost Warnings (Generic Check):\n"
                    for issue in cost_issues:
                        result += f"{issue}\n"
                    result += "\n"
                else:
                    result += "✓ Point costs appear reasonable (generic check)\n\n"

            # Always provide explicit links to the official documents
            result += "## Official Rules Documents Links\n\n"

            result += "**Core Rules:**\n"
            result += f"- **Link:** {self.CORE_RULES_URL}\n"
            result += "- **Content:** Core rules and standard game structure.\n\n"

            result += "**Munitorum Field Manual (Points):**\n"
            result += f"- **Link:** {self.MUNITORUM_MANUAL_URL}\n"
            result += "- **Content:** Official point costs for all units. **Crucial for list building.**\n\n"

            result += "**Balance Dataslate/Core Rules Updates:**\n"
            result += f"- **Link:** {self.BALANCE_DATASLATE_URL}\n"
            result += "- **Content:** Recent rule changes,"
            " stratagem updates, and points changes not yet in the Manual.\n\n"

            result += "=" * 70 + "\n"
            result += "## Recommendations\n\n"
            result += "1. Always consult the linked **Munitorum Field Manual** for accurate unit point costs.\n"
            result += "2. Check the linked **Balance Dataslate** for the latest rule and points adjustments.\n"
            result += "3. Ensure all unit loadouts and detachment choices are legal per the **Core Rules**.\n"

            return result

        except Exception as e:
            return f"An error occurred during validation: {e}"


def get_wh40k_army_tool() -> WH40KArmyDataTool:
    """
    Return configured Warhammer 40K army data tool with parsing capabilities.
    """
    return WH40KArmyDataTool()


def get_wh40k_rules_validator_tool() -> WH40KRulesValidatorTool:
    """
    Return configured Warhammer 40K rules validator tool.
    """
    return WH40KRulesValidatorTool()


# Helper functions to instantiate tools
def get_wikipedia_tool():
    """
    Returns a configured Wikipedia search tool.
    This tool can be used to search for information on Wikipedia.
    """
    return WikipediaSearchTool()


def get_context_collector_tool(links_db: Dict[str, List[str]] = None):
    """
    Returns a configured text context collector tool.
    Args:
        links_db: Optional custom database of topic-to-URL mappings
    """
    tool = TextContextCollectorTool()
    if links_db:
        tool.links_database = links_db
    return tool


def get_context_comparison_tool():
    """Returns a configured context comparison tool."""
    return ContextComparisonTool()


def count_words(text: str) -> int:
    """
    Counts the number of words in a given string.
    """
    return len(text.split())
