// Codenames AI Agent Schemas and Functions
// This file defines the BAML types and functions for LLM-based Codenames agents

// ============================================================================
// SCHEMA DEFINITIONS
// ============================================================================

class HintResponse {
  word string @description("One-word hint (no spaces, not on the board)")
  count int @description("Number of words this hint relates to (1-9)")
  reasoning string @description("Brief explanation of strategy and word associations")
}

class GuessResponse {
  guesses string[] @description("List of words to guess, ordered by confidence (most confident first)")
  reasoning string @description("Brief explanation of why these words relate to the hint")
}

// ============================================================================
// HINT GIVER FUNCTION (SPYMASTER)
// ============================================================================

function GiveHint(
  team: string @description("Team color: 'blue' or 'red'"),
  my_words: string[] @description("Your team's unrevealed words that need to be guessed"),
  opponent_words: string[] @description("Opponent's unrevealed words to avoid"),
  neutral_words: string[] @description("Neutral unrevealed words to avoid"),
  bomb_word: string @description("The bomb word - NEVER hint at this or you lose!"),
  revealed_words: string[] @description("Already revealed words (for context)")
) -> HintResponse {
  client GPT4oMini  // Default client - can be overridden at runtime

  prompt #"
    You are playing Codenames as the {{ team | upper }} team's spymaster.

    YOUR GOAL: Give a one-word hint and a number to help your teammate guess your team's words.

    YOUR TEAM'S WORDS (need to be guessed):
    {{ my_words | join(', ') }}

    OPPONENT'S WORDS (avoid these):
    {{ opponent_words | join(', ') }}

    NEUTRAL WORDS (avoid these):
    {{ neutral_words | join(', ') }}

    BOMB WORD (NEVER hint at this):
    {{ bomb_word }}

    {% if revealed_words | length > 0 %}
    ALREADY REVEALED:
    {{ revealed_words | join(', ') }}
    {% endif %}

    RULES:
    1. Give a ONE-WORD hint (no spaces, no words from the board)
    2. Give a NUMBER indicating how many of your words relate to this hint
    3. Your hint should connect multiple of your words if possible
    4. Avoid hints that could lead to opponent words, neutral words, or the BOMB
    5. Be strategic - think about semantic associations, categories, and relationships

    STRATEGY TIPS:
    - Look for semantic clusters (e.g., "animal" for dog, cat, mouse)
    - Consider word relationships (e.g., "royalty" for king, queen, crown)
    - Balance safety vs. aggressiveness based on game state
    - Avoid risky hints that could lead to the bomb

    {{ ctx.output_format }}
  "#
}

// ============================================================================
// GUESSER FUNCTION (FIELD OPERATIVE)
// ============================================================================

function MakeGuesses(
  team: string @description("Team color: 'blue' or 'red'"),
  hint_word: string @description("The hint word given by your spymaster"),
  hint_count: int @description("Number of words the hint relates to"),
  board_words: string[] @description("All words currently on the board"),
  revealed_words: string[] @description("Already revealed words (don't guess these)")
) -> GuessResponse {
  client GPT4oMini  // Default client - can be overridden at runtime

  prompt #"
    You are playing Codenames as the {{ team | upper }} team's field operative.

    YOUR HINT: "{{ hint_word }}" ({{ hint_count }})
    This means your spymaster wants you to guess {{ hint_count }} word(s) related to "{{ hint_word }}".

    WORDS ON THE BOARD (unrevealed):
    {% set unrevealed = [] %}
    {% for word in board_words %}
      {% if word not in revealed_words %}
        {% set unrevealed = unrevealed + [word] %}
      {% endif %}
    {% endfor %}
    {{ unrevealed | join(', ') }}

    {% if revealed_words | length > 0 %}
    ALREADY REVEALED (don't guess these):
    {{ revealed_words | join(', ') }}
    {% endif %}

    YOUR TASK:
    1. Identify which unrevealed words relate to the hint "{{ hint_word }}"
    2. Return up to {{ hint_count }} words (you can guess fewer if unsure)
    3. Order them by confidence (most confident first)
    4. You can optionally guess {{ hint_count + 1 }} words if you want to use a previous hint

    IMPORTANT:
    - Only guess words from the unrevealed list above
    - If you guess wrong, your turn ends immediately
    - If you hit the bomb, your team loses the game
    - Be thoughtful - quality over quantity
    - It's better to guess fewer words confidently than to guess risky words

    STRATEGY:
    - Think about semantic relationships and word associations
    - Consider multiple meanings of the hint word
    - Rank words by how strongly they relate to the hint
    - If uncertain about a word, leave it out

    {{ ctx.output_format }}
  "#
}
