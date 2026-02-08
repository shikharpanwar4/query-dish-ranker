# Build dish catalog + (query, dish_name) pairs; optional LLM JSONs. Writes dishes.csv, train.csv, val.csv. Run: python data/prepare_data.py

import os
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # data/
RAW_INDIAN_FOOD = BASE_DIR / "indian_food.csv"
RAW_SWIGGY = BASE_DIR / "swiggy_cleaned.csv"
RAW_SWIGGY_BANGALORE = BASE_DIR / "swiggy_all_menus_bangalore.csv"
RAW_ARCHANAS = BASE_DIR / "IndianFoodDatasetCSV.csv"  # ~6000 recipes
LLM_DIR = BASE_DIR / "raw"                          # for LLM-generated JSONs
PROCESSED_DIR = BASE_DIR / "processed"

# Category values from Swiggy Bangalore to skip (promo/UI, not queryable)
SWIGGY_BANGALORE_CATEGORY_SKIP = frozenset({"recommended", "value scoops", "central bangalore", "mcsaver combos (2 pc meals)", "freshly scooped tubs"})

# Use only Indian Food 101 (255 dishes). Set True to add Archana's Kitchen again.
USE_ARCHANAS = False

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
random.seed(42)

DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')
STOP_WORDS = {"and", "with", "the", "of", "in", "a", "or", "for", "to", "on", "at", "by", "&", "-", "|", "style", "type", "special", "recipe"}

# Romanized Hindi words: if any in query, treat as Hinglish (skip reorder / UPPER to reduce noise)
HINGLISH_WORDS = frozenset({
    "kuch", "chahiye", "wala", "banao", "meetha", "teekha", "khaana", "bhai", "accha", "karna", "hai",
    "masaledar", "khana", "meetha", "sa", "jaisa", "chahiye", "wale", "bana", "khao", "try",
})

# --- Stage 1: Load & clean catalog ---

def load_indian_food(path: str) -> list[dict]:
    """Load indian_food.csv; -1 → None, ingredients → list, lowercase."""
    dishes = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # --- clean name ---
            name = row["name"].strip()
            if not name:
                continue

            # --- clean ingredients → list ---
            raw_ingredients = row.get("ingredients", "")
            ingredients = [
                ing.strip().lower()
                for ing in raw_ingredients.split(",")
                if ing.strip() and ing.strip() != "-1"
            ]

            # --- clean categorical fields (replace -1 with None) ---
            def clean_field(val: str):
                val = val.strip().lower()
                return val if val not in ("-1", "", "unknown") else None

            dishes.append({
                "name": name,
                "name_lower": name.lower(),
                "ingredients": ingredients,
                "diet": clean_field(row.get("diet", "")),
                "flavor": clean_field(row.get("flavor_profile", "")),
                "course": clean_field(row.get("course", "")),
                "state": clean_field(row.get("state", "")),
                "region": clean_field(row.get("region", "")),
                "style": None,
                "category": None,
                "source": "indian_food",
            })
    return dishes


def extract_swiggy_food_categories(path: str) -> list[str]:
    """Unique food_type values from swiggy CSV (e.g. Biryani, North Indian)."""
    categories = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            food_type = row.get("food_type", "")
            for cat in food_type.split(","):
                cat = cat.strip()
                if cat and cat.lower() not in ("not_available", ""):
                    categories.add(cat)
    # Return sorted for determinism
    return sorted(categories)


def load_swiggy_bangalore_dishes(path: Path) -> list[dict]:
    """Load swiggy_all_menus_bangalore.csv; dedupe by normalized dish name; return list with name, category, source=swiggy_bangalore, empty metadata."""
    if not path.exists():
        return []
    name_to_row: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("Dish Name") or "").strip()
            if not name or len(name) < 2:
                continue
            category = (row.get("Category") or "").strip()
            name_lower = name.lower()
            if name_lower not in name_to_row:
                name_to_row[name_lower] = {
                    "name": name,
                    "name_lower": name_lower,
                    "ingredients": [],
                    "diet": None,
                    "flavor": None,
                    "course": None,
                    "state": None,
                    "region": None,
                    "style": None,
                    "category": category.lower() if category else None,
                    "source": "swiggy_bangalore",
                }
    return list(name_to_row.values())


def extract_swiggy_bangalore_categories(path: Path) -> list[str]:
    """Distinct Category values from swiggy_all_menus_bangalore; exclude promo-like labels."""
    if not path.exists():
        return []
    categories = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = (row.get("Category") or "").strip().lower()
            if cat and cat not in SWIGGY_BANGALORE_CATEGORY_SKIP:
                categories.add(cat)
    return sorted(categories)


def build_ingredient_index(dishes: list[dict]) -> dict[str, list[str]]:
    """Ingredient → list of dish names (for query generation)."""
    index = defaultdict(list)
    for dish in dishes:
        for ing in dish["ingredients"]:
            # Normalize multi-word ingredients to their key word
            # "cottage cheese" → keep as-is (we'll query both ways)
            index[ing].append(dish["name"])
    return dict(index)


def build_metadata_index(dishes: list[dict]) -> dict:
    """Group dishes by diet/flavor/course/region/state/category for lookup."""
    index = {
        "diet": defaultdict(list),
        "flavor": defaultdict(list),
        "course": defaultdict(list),
        "region": defaultdict(list),
        "state": defaultdict(list),
        "category": defaultdict(list),
    }
    for dish in dishes:
        for key in index:
            val = dish.get(key)
            if val:
                index[key][val].append(dish["name"])
    return {k: dict(v) for k, v in index.items()}


# --- Archana's Kitchen dataset helpers --------------------------------

def clean_recipe_name(name: str) -> str:
    """Strip ' - ...', parentheticals, 'Recipe...'; cap at 8 words (Archana's format)."""
    if " - " in name:
        name = name.split(" - ")[0].strip()
    if " | " in name:
        name = name.split(" | ")[0].strip()
    name = re.sub(r'\s*\([^)]*\)', '', name)
    name = re.sub(r'\s+Recipes?\b.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^Recipes?\s+', '', name, flags=re.IGNORECASE)
    name = re.sub(r'Recipes?$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\([^)]*$', '', name)
    words = name.split()
    if len(words) > 8:
        name = " ".join(words[:8])
    return name.strip()


# Patterns that indicate a non-dish entry (tutorials, tips, etc.)
NON_DISH_PATTERNS = re.compile(
    r'^(?:How [Tt]o |Health Benefits|Tips [Ff]or |What [Ii]s |'
    r'Ways [Tt]o |Guide [Tt]o |Difference [Bb]etween |'
    r'Benefits [Oo]f |Importance [Oo]f )',
)


def clean_ingredient_str(raw: str) -> list[str]:
    """Parse Archana's ingredient string: drop quantities, parentheticals, prep notes → list of nouns."""
    if not raw:
        return []

    ingredients = []
    for part in raw.split(","):
        part = part.strip()
        if not part or DEVANAGARI_RE.search(part):
            continue
        cleaned = re.sub(r'\s*\([^)]*\)', '', part)
        cleaned = cleaned.split(' - ')[0].strip()
        cleaned = re.sub(
            r'^[\d\s/.½¼¾⅓⅔-]*'
            r'(?:cups?|tablespoons?|teaspoons?|tbsp|tsp|grams?|g\b|kg|ml|'
            r'liters?|litres?|inch(?:es)?|sprigs?|cloves?|pinch(?:es)?|bunch(?:es)?|'
            r'pieces?|slices?|small|medium|large|whole|numbers?|handful|'
            r'sticks?|pods?|sheets?|leaves|drops?|quarts?|pounds?|oz)\s*',
            '', cleaned, flags=re.IGNORECASE,
        )
        cleaned = re.sub(r'^[\d\s/.½¼¾⅓⅔-]+', '', cleaned)
        cleaned = re.sub(
            r'\s+(?:chopped|sliced|grated|minced|diced|crushed|ground|'
            r'powdered|dried|fresh|finely|roughly|thinly|optional|'
            r'for garnish|for cooking|for frying|for greasing)\s*$',
            '', cleaned, flags=re.IGNORECASE,
        )
        cleaned = cleaned.strip().lower()

        if cleaned and len(cleaned) >= 2:
            ingredients.append(cleaned)

    return ingredients


def load_archanas_kitchen(path: str) -> list[dict]:
    """Load Archana's CSV: TranslatedRecipeName, filter Devanagari, clean names, map cuisine→region, dedupe."""
    cuisine_to_region = {
        "north indian recipes": "north",
        "south indian recipes": "south",
        "bengali recipes": "east",
        "gujarati recipes": "west",
        "maharashtrian recipes": "west",
        "rajasthani": "west",
        "andhra": "south",
        "chettinad": "south",
        "kerala recipes": "south",
        "karnataka": "south",
        "tamil nadu": "south",
        "hyderabadi": "south",
        "kashmiri": "north",
        "punjabi": "north",
        "lucknowi": "north",
        "awadhi": "north",
        "mughlai": "north",
        "goan recipes": "west",
        "assamese": "north east",
        "mangalorean": "south",
        "udupi": "south",
        "konkan": "west",
    }

    # --- Normalize diet values ---
    diet_map = {
        "vegetarian": "vegetarian",
        "non vegeterian": "non vegetarian",     # fix typo in dataset
        "non vegetarian": "non vegetarian",
        "eggetarian": "vegetarian",
        "vegan": "vegetarian",
        "high protein vegetarian": "vegetarian",
        "high protein non vegetarian": "non vegetarian",
        "diabetic friendly": "vegetarian",
        "no onion no garlic (sattvic)": "vegetarian",
        "sugar free diet": "vegetarian",
        "gluten free": None,
    }

    # --- Normalize course values ---
    course_map = {
        "side dish": "snack",
        "main course": "main course",
        "lunch": "main course",
        "dinner": "main course",
        "snack": "snack",
        "appetizer": "snack",
        "dessert": "dessert",
        "south indian breakfast": "snack",
        "indian breakfast": "snack",
        "north indian breakfast": "snack",
        "breakfast": "snack",
        "one pot dish": "main course",
        "brunch": "main course",
        "world breakfast": "snack",
    }

    dishes = []
    seen_names = set()
    skipped_hindi = 0
    skipped_dup = 0
    skipped_nondish = 0
    skipped_other = 0

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            translated_name = row.get("TranslatedRecipeName", "").strip()
            if not translated_name:
                skipped_other += 1
                continue

            # Filter out entries with Devanagari characters
            if DEVANAGARI_RE.search(translated_name):
                skipped_hindi += 1
                continue

            # Clean the name
            name = clean_recipe_name(translated_name)
            if not name or len(name) < 3:
                skipped_other += 1
                continue

            # Safety: re-check cleaned name for stray Devanagari
            if DEVANAGARI_RE.search(name):
                skipped_hindi += 1
                continue

            # A1: Filter non-dish entries (tutorials, health tips, etc.)
            if NON_DISH_PATTERNS.search(name):
                skipped_nondish += 1
                continue

            # Deduplicate within this dataset
            name_lower = name.lower()
            if name_lower in seen_names:
                skipped_dup += 1
                continue
            seen_names.add(name_lower)

            # Parse ingredients (use Translated/English version)
            raw_ingredients = row.get("TranslatedIngredients", "")
            ingredients = clean_ingredient_str(raw_ingredients)

            # Parse and normalize metadata
            cuisine = (row.get("Cuisine", "") or "").strip().lower()
            course_raw = (row.get("Course", "") or "").strip().lower()
            diet_raw = (row.get("Diet", "") or "").strip().lower()

            region = cuisine_to_region.get(cuisine)
            diet = diet_map.get(diet_raw)
            course = course_map.get(course_raw, course_raw if course_raw else None)

            # Store raw cuisine for category query generation (A5)
            cuisine_clean = cuisine.replace(" recipes", "").strip() if cuisine else None

            dishes.append({
                "name": name,
                "name_lower": name_lower,
                "ingredients": ingredients,
                "diet": diet,
                "flavor": None,
                "course": course,
                "state": None,
                "region": region,
                "cuisine": cuisine_clean,
                "source": "archanas",
            })

    print(f"  Skipped {skipped_hindi} entries with Hindi/Devanagari text")
    print(f"  Skipped {skipped_dup} duplicate names within dataset")
    print(f"  Skipped {skipped_nondish} non-dish entries (tutorials/tips)")
    print(f"  Skipped {skipped_other} entries with empty/short names")
    return dishes


# --- Stage 2: Programmatic query-dish pairs ---

def generate_exact_match_pairs(dishes: list[dict]) -> list[tuple[str, str]]:
    """Query = dish name + lowercase variant."""
    pairs = []
    for dish in dishes:
        name = dish["name"]
        pairs.append((name, name))                   # exact case
        pairs.append((name.lower(), name))            # lowercase
    return pairs


def generate_partial_name_pairs(dishes: list[dict]) -> list[tuple[str, str]]:
    """Partial dish name queries (drop one word, first word only, last word only); 2+ word names only."""
    pairs = []
    for dish in dishes:
        words = dish["name"].split()
        if len(words) < 2:
            continue

        # Drop each word one at a time
        for i in range(len(words)):
            partial = " ".join(words[:i] + words[i + 1:])
            if partial.strip():
                pairs.append((partial.lower(), dish["name"]))

        # First word only (for 3+ word names)
        if len(words) >= 3:
            pairs.append((words[0].lower(), dish["name"]))

        # Last word only
        pairs.append((words[-1].lower(), dish["name"]))

    return pairs


def generate_ingredient_query_pairs(
    dishes: list[dict],
    ingredient_index: dict[str, list[str]],
    max_dishes_per_query: int = 3,
) -> list[tuple[str, str]]:
    """Ingredient-based queries; only keep pairs where ingredient appears in dish name (learnable signal)."""
    templates = [
        "{ingredient}",
        "{ingredient} dish",
        "{ingredient} food",
        "something with {ingredient}",
        "{ingredient} wala",
    ]

    # Skip very generic ingredients that appear in nearly everything.
    skip_ingredients = {
        "salt", "oil", "water", "sugar", "ghee", "mustard oil", "turmeric",
        "turmeric powder", "red chilli", "red chilli powder", "red chili powder",
        "green chilli", "green chillies", "ginger", "garlic", "onion", "onions",
        "tomato", "tomatoes", "cumin", "cumin seeds", "coriander",
        "coriander powder", "bay leaf", "cinnamon", "cinnamon stick",
        "cardamom", "rock salt", "black salt", "chili powder", "mustard seeds",
        "curry leaves", "coconut oil", "olive oil", "sesame oil", "sunflower oil",
        "garam masala powder", "black pepper", "black pepper powder",
        "asafoetida", "baking powder", "baking soda", "butter", "cream",
        "flour", "all purpose flour", "rice", "lemon juice", "vinegar",
    }

    pairs = []
    skipped_no_overlap = 0
    for ingredient, dish_names in ingredient_index.items():
        if ingredient in skip_ingredients:
            continue
        if len(dish_names) > 100:
            continue

        ing_words = set(ingredient.lower().split())

        # Sample a limited number of target dishes per ingredient query
        sampled_dishes = dish_names
        if len(dish_names) > max_dishes_per_query:
            sampled_dishes = random.sample(dish_names, max_dishes_per_query)

        for dish_name in sampled_dishes:
            dish_words = set(dish_name.lower().split())
            if not (ing_words & dish_words):
                skipped_no_overlap += 1
                continue

            for template in templates:
                query = template.format(ingredient=ingredient)
                pairs.append((query, dish_name))

    if skipped_no_overlap:
        print(f"    (filtered {skipped_no_overlap} ingredient→dish pairs with no name overlap)")

    return pairs


def generate_category_query_pairs(
    dishes: list[dict],
    metadata_index: dict,
    swiggy_categories: list[str],
    max_dishes_per_query: int = 3,
) -> list[tuple[str, str]]:
    """Category/attribute combos (diet+course, flavor+course, etc.) + region food + swiggy; limit dishes per query."""
    pairs = []
    combos = [
        ("diet", "course"),
        ("flavor", "course"),
        ("region", "course"),
        ("diet", "flavor"),
    ]
    combo_groups = defaultdict(list)  # query → list of matching dishes
    for dish in dishes:
        for field_a, field_b in combos:
            val_a = dish.get(field_a)
            val_b = dish.get(field_b)
            if val_a and val_b:
                query = f"{val_a} {val_b}"
                combo_groups[query].append(dish["name"])

    for query, dish_names in combo_groups.items():
        sampled = dish_names if len(dish_names) <= max_dishes_per_query else random.sample(dish_names, max_dishes_per_query)
        for dish_name in sampled:
            pairs.append((query, dish_name))

    for region_val, dish_names in metadata_index["region"].items():
        for suffix in ["food", "dish", "cuisine"]:
            query = f"{region_val} indian {suffix}"
            sampled = dish_names
            if len(dish_names) > max_dishes_per_query:
                sampled = random.sample(dish_names, max_dishes_per_query)
            for dish_name in sampled:
                pairs.append((query, dish_name))

    category_to_field = {
        "north indian": ("region", "north"),
        "south indian": ("region", "south"),
        "desserts": ("course", "dessert"),
        "snacks": ("course", "snack"),
        "sweets": ("flavor", "sweet"),
    }
    for cat in swiggy_categories:
        cat_lower = cat.lower()
        if cat_lower in category_to_field:
            field, value = category_to_field[cat_lower]
            dish_names = metadata_index.get(field, {}).get(value, [])
            sampled = dish_names
            if len(dish_names) > max_dishes_per_query:
                sampled = random.sample(dish_names, max_dishes_per_query)
            for dish_name in sampled:
                pairs.append((cat_lower, dish_name))

    # Category from metadata (e.g. Swiggy Bangalore: desserts, appetizer, beverages)
    for cat_val, dish_names in metadata_index.get("category", {}).items():
        if not cat_val or len(dish_names) > 200:
            continue
        sampled = dish_names if len(dish_names) <= max_dishes_per_query else random.sample(dish_names, max_dishes_per_query)
        for dish_name in sampled:
            pairs.append((cat_val, dish_name))

    return pairs


def generate_cuisine_query_pairs(dishes: list[dict], max_dishes_per_query: int = 3) -> list[tuple[str, str]]:
    """Cuisine + course and cuisine + suffix (uses Archana's 'cuisine' field when present)."""
    pairs = []
    combo_groups = defaultdict(list)
    for dish in dishes:
        cuisine = dish.get("cuisine")
        course = dish.get("course")
        if cuisine and course:
            # "south indian" + "dessert" → "south indian dessert"
            query = f"{cuisine} {course}"
            combo_groups[query].append(dish["name"])

    for query, dish_names in combo_groups.items():
        if len(dish_names) > 50:
            continue
        sampled = dish_names
        if len(dish_names) > max_dishes_per_query:
            sampled = random.sample(dish_names, max_dishes_per_query)
        for dish_name in sampled:
            pairs.append((query, dish_name))

    cuisine_groups = defaultdict(list)
    for dish in dishes:
        cuisine = dish.get("cuisine")
        if cuisine:
            cuisine_groups[cuisine].append(dish["name"])

    for cuisine_val, dish_names in cuisine_groups.items():
        if cuisine_val in ("indian",) or len(dish_names) > 200:
            continue
        for suffix in ["food", "dish", "cuisine"]:
            query = f"{cuisine_val} {suffix}"
            sampled = dish_names
            if len(dish_names) > max_dishes_per_query:
                sampled = random.sample(dish_names, max_dishes_per_query)
            for dish_name in sampled:
                pairs.append((query, dish_name))

    return pairs


def generate_attribute_query_pairs(dishes: list[dict]) -> list[tuple[str, str]]:
    """Attribute (diet/flavor/course) + dish-name word, e.g. 'spicy paneer', 'veg biryani'; overlap in name."""
    diet_attrs = {
        "vegetarian": ["veg"],
        "non vegetarian": ["non veg"],
    }
    flavor_attrs = {
        "sweet": ["sweet", "meetha"],
        "spicy": ["spicy", "teekha", "masaledar"],
        "bitter": ["bitter"],
        "sour": ["sour", "khatta"],
    }
    course_attrs = {
        "dessert": ["dessert", "mithai"],
        "snack": ["snack", "nashta"],
        "main course": ["lunch", "dinner"],
    }

    pairs = []
    for dish in dishes:
        name_words = [w for w in dish["name"].split() if w.lower() not in STOP_WORDS and len(w) >= 3]
        if not name_words:
            continue
        pick = sorted(name_words, key=lambda w: -len(w))[:2]
        attrs = []
        diet = dish.get("diet")
        if diet and diet in diet_attrs:
            attrs.extend(diet_attrs[diet])
        flavor = dish.get("flavor")
        if flavor and flavor in flavor_attrs:
            attrs.extend(flavor_attrs[flavor])
        course = dish.get("course")
        if course and course in course_attrs:
            attrs.extend(course_attrs[course])

        if not attrs:
            continue

        for attr in attrs:
            for word in pick:
                query = f"{attr} {word.lower()}"
                pairs.append((query, dish["name"]))

    return pairs


def generate_occasion_query_pairs(dishes: list[dict], max_dishes_per_query: int = 3) -> list[tuple[str, str]]:
    """Occasion phrases (party snack, sweet dessert) → dishes with matching course and name-word overlap."""
    occasion_rules = [
        ("party snack", "snack", {"snack", "chaat", "pakora", "puri", "samosa", "vada", "idli", "dosa", "bhel", "sev", "papdi", "bajji", "bonda"}),
        ("evening snack", "snack", {"snack", "chaat", "pakora", "samosa", "vada", "idli", "dosa", "bhel", "sev"}),
        ("sweet dessert", "dessert", {"dessert", "sweet", "mithai", "halwa", "kheer", "ladoo", "barfi", "gulab", "jamun", "rasmalai", "kulfi", "cake", "pie", "pastry", "meetha"}),
        ("healthy dessert", "dessert", {"dessert", "halwa", "kheer", "fruit", "dates", "nuts"}),
    ]

    pairs = []
    for query_phrase, course_val, required_words in occasion_rules:
        candidates = [
            d for d in dishes
            if d.get("course") == course_val
            and any(w.lower() in required_words for w in d["name"].split())
        ]
        if not candidates:
            continue
        sampled = candidates if len(candidates) <= max_dishes_per_query else random.sample(candidates, max_dishes_per_query)
        for dish in sampled:
            pairs.append((query_phrase, dish["name"]))

    return pairs


def generate_hinglish_query_pairs(dishes: list[dict]) -> list[tuple[str, str]]:
    """Hinglish templates with dish-name word: 'kuch X chahiye', 'X banao', etc."""
    templates = [
        "kuch {} chahiye",
        "meetha {}",
        "spicy {}",
        "{} banao",
        "{} wala",
    ]

    pairs = []
    for dish in dishes:
        name_words = [w for w in dish["name"].split() if w.lower() not in STOP_WORDS and len(w) >= 3]
        if not name_words:
            continue
        word = max(name_words, key=len).lower()
        for t in templates:
            query = t.format(word)
            pairs.append((query, dish["name"]))

    return pairs


def generate_synthetic_llm_style_pairs(dishes: list[dict], queries_per_dish: int = 4) -> list[tuple[str, str]]:
    """LLM-style natural queries per dish (Hinglish, occasion, attribute) using name + metadata."""
    pairs = []
    for dish in dishes:
        name = dish["name"]
        words = [w for w in name.split() if w.lower() not in STOP_WORDS and len(w) >= 3]
        if not words:
            continue
        word = random.choice(words).lower()
        if len(words) >= 2:
            word2 = max(words, key=len).lower()
            if word2 != word:
                word = random.choice([word, word2])

        region = (dish.get("region") or "").strip()
        state = (dish.get("state") or "").strip()
        course = (dish.get("course") or "").strip()
        flavor = (dish.get("flavor") or "").strip()
        diet = (dish.get("diet") or "").strip()
        templates = []
        templates.append(f"kuch {word} chahiye")
        templates.append(f"{word} try karna hai")
        templates.append(f"something {word}")
        if flavor == "sweet":
            templates.append(f"meetha {word}")
            templates.append("kuch meetha chahiye")
        if flavor == "spicy":
            templates.append(f"spicy {word}")
            templates.append("kuch teekha khana hai")
        if course:
            templates.append(f"{course} idea")
            templates.append(f"light {course}")
        if region:
            templates.append(f"{region} style food")
            templates.append(f"{region} khaana")
        if state:
            templates.append(f"{state} dish")
        if diet:
            templates.append(f"{diet} {word}")
        if course and word:
            templates.append(f"{course} {word}")
        templates.append(f"want to try {word}")
        templates.append(f"something like {word}")

        chosen = list(dict.fromkeys(templates))  # dedup preserving order
        random.shuffle(chosen)
        for q in chosen[:queries_per_dish]:
            pairs.append((q, name))

    return pairs


# --- Stage 3: Augmentation ---

def inject_typo(text: str, num_typos: int = 1) -> str:
    """Swap/delete/duplicate char in a word (4+ chars); one or more typos."""
    words = text.split()
    if not words:
        return text

    for _ in range(num_typos):
        long_words = [(i, w) for i, w in enumerate(words) if len(w) >= 4]
        if not long_words:
            break
        idx, word = random.choice(long_words)
        word_chars = list(word)

        op = random.choice(["swap", "delete", "duplicate"])
        pos = random.randint(1, len(word_chars) - 2)

        if op == "swap" and pos < len(word_chars) - 1:
            word_chars[pos], word_chars[pos + 1] = word_chars[pos + 1], word_chars[pos]
        elif op == "delete":
            word_chars.pop(pos)
        elif op == "duplicate":
            word_chars.insert(pos, word_chars[pos])

        words[idx] = "".join(word_chars)

    return " ".join(words)


def _is_hinglish(query: str) -> bool:
    """True if query contains Romanized Hindi tokens (avoid reorder/UPPER to reduce noise)."""
    words = set(w.lower() for w in query.split())
    return bool(words & HINGLISH_WORDS)


def augment_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Deterministic case variants; deterministic reorder for short English-only; limited typos (reproducible via seed)."""
    augmented = []
    for query, dish in pairs:
        # Deterministic case: title for all; upper only for English (Hinglish UPPER adds noise)
        title_q = query.title()
        if title_q != query:
            augmented.append((title_q, dish))
        if not _is_hinglish(query):
            upper_q = query.upper()
            if upper_q != query:
                augmented.append((upper_q, dish))

    # Deterministic reorder: only short (2–4 word) English queries; add reversed word order
    multi_word_english = [(q, d) for q, d in pairs if 2 <= len(q.split()) <= 4 and not _is_hinglish(q)]
    for query, dish in multi_word_english:
        words = query.split()
        rev = " ".join(reversed(words))
        if rev != query:
            augmented.append((rev, dish))

    # Limited typos (reproducible): 10% of pairs with a 4+ char word, skip Hinglish
    typo_candidates = [(q, d) for q, d in pairs if not _is_hinglish(q) and any(len(w) >= 4 for w in q.split())]
    typo_count = min(int(len(pairs) * 0.10), len(typo_candidates))
    for query, dish in random.sample(typo_candidates, k=typo_count):
        augmented.append((inject_typo(query, num_typos=1), dish))

    return augmented


# --- Stage 4: LLM pairs from JSON ---

def _build_keyword_to_dishes(dishes: list[dict]) -> dict[str, list[str]]:
    """Build keyword (lower) -> list of dish names for resolving expected_keywords."""
    index = defaultdict(list)
    for d in dishes:
        name = d.get("name") or ""
        name_lower = name.lower()
        if name_lower:
            index[name_lower].append(name)
        for w in name.split():
            w = w.lower()
            if len(w) >= 2 and w not in STOP_WORDS and name not in (index.get(w) or []):
                if name not in index[w]:
                    index[w].append(name)
        for ing in (d.get("ingredients") or []):
            if isinstance(ing, str):
                w = ing.strip().lower()
            else:
                w = str(ing).strip().lower()
            if len(w) >= 2 and w not in STOP_WORDS:
                if name not in index[w]:
                    index[w].append(name)
        cat = (d.get("category") or "").strip().lower()
        if cat and name not in index.get(cat, []):
            index[cat].append(name)
    return dict(index)


def _resolve_keywords_to_dish(
    expected_keywords: list[str],
    keyword_to_dishes: dict[str, list[str]],
    valid_dishes: set[str],
) -> str | None:
    """Return one dish name that matches expected_keywords (prefer name match); None if none."""
    candidates = []
    for kw in (expected_keywords or []):
        k = (kw or "").strip().lower()
        if not k:
            continue
        for dish in keyword_to_dishes.get(k, []):
            if dish.lower() in valid_dishes:
                candidates.append(dish)
    if not candidates:
        return None
    # Prefer dish that appears most (multiple keywords pointing to it)
    counts = Counter(candidates)
    return counts.most_common(1)[0][0]


def load_llm_pairs(
    llm_dir: str,
    valid_dishes: set[str] | None = None,
    dishes: list[dict] | None = None,
) -> list[tuple[str, str]]:
    """Load llm_queries*.json. Items: {query, dish} or {query, category, expected_keywords}. Resolve keywords to catalog."""
    dish_aliases = {
        "Halwa (use Gajar ka halwa)": "Gajar ka halwa",
        "Kebabs (use Chicken Tikka)": "Chicken Tikka",
        "Halwa": "Gajar ka halwa",
        "Kebabs": "Chicken Tikka",
    }
    keyword_to_dishes = _build_keyword_to_dishes(dishes) if dishes else {}
    valid = valid_dishes or set()

    pairs = []
    skipped_dish = 0
    skipped_quality = 0
    skipped_no_match = 0
    llm_path = Path(llm_dir)
    if not llm_path.exists():
        return pairs

    for json_file in sorted(llm_path.glob("llm_queries*.json")):
        print(f"  Loading LLM pairs from {json_file.name}...")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            query = (item.get("query") or "").strip()
            if not query or len(query.split()) > 10:
                skipped_quality += 1
                continue

            dish = (item.get("dish") or "").strip()
            if dish:
                dish = dish_aliases.get(dish, dish)
                if valid and dish.lower() not in valid:
                    skipped_dish += 1
                    continue
                if NON_DISH_PATTERNS.search(dish):
                    skipped_quality += 1
                    continue
                pairs.append((query, dish))
                continue

            # Format: {query, category, expected_keywords}
            expected_keywords = item.get("expected_keywords")
            if expected_keywords is not None and keyword_to_dishes and valid:
                resolved = _resolve_keywords_to_dish(
                    expected_keywords if isinstance(expected_keywords, list) else [expected_keywords],
                    keyword_to_dishes,
                    valid,
                )
                if resolved:
                    pairs.append((query, resolved))
                else:
                    skipped_no_match += 1
            else:
                skipped_quality += 1

    if skipped_dish:
        print(f"  Skipped {skipped_dish} pairs with unrecognized dish names")
    if skipped_quality:
        print(f"  Skipped {skipped_quality} low-quality pairs (long query / no dish or keywords)")
    if skipped_no_match:
        print(f"  Skipped {skipped_no_match} keyword-form pairs with no catalog match")

    return pairs


# --- Stage 5: Dedupe, split, save ---

def deduplicate_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Dedupe (query, dish) by (q.lower(), d.lower()); preserve order."""
    seen = set()
    unique = []
    for pair in pairs:
        key = (pair[0].lower(), pair[1].lower())
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    return unique


def split_train_val(
    pairs: list[tuple[str, str]],
    val_fraction: float = 0.1,
    dish_to_source: dict[str, str] | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split by dish (no leakage); optional stratify by source."""
    dish_to_pairs = defaultdict(list)
    for query, dish in pairs:
        dish_to_pairs[dish].append((query, dish))

    all_dishes = list(dish_to_pairs.keys())

    if dish_to_source:
        by_source = defaultdict(list)
        for d in all_dishes:
            src = dish_to_source.get(d, "unknown")
            by_source[src].append(d)
        val_dishes = set()
        for src, dishes_list in by_source.items():
            random.shuffle(dishes_list)
            n_val = max(1, int(len(dishes_list) * val_fraction))
            val_dishes.update(dishes_list[:n_val])
    else:
        random.shuffle(all_dishes)
        val_count = max(1, int(len(all_dishes) * val_fraction))
        val_dishes = set(all_dishes[:val_count])

    train_pairs = []
    val_pairs = []
    for dish, dish_pairs in dish_to_pairs.items():
        if dish in val_dishes:
            val_pairs.extend(dish_pairs)
        else:
            train_pairs.extend(dish_pairs)

    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    return train_pairs, val_pairs


def save_pairs(pairs: list[tuple[str, str]], path: str):
    """Save (query, dish_name) pairs as CSV."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "dish_name"])
        for query, dish in pairs:
            writer.writerow([query, dish])
    print(f"  Saved {len(pairs):,} pairs → {path}")


def save_dish_catalog(dishes: list[dict], path: str):
    """Write dishes to CSV (name, ingredients, diet, flavor, course, state, region, style, category, source)."""
    fieldnames = ["name", "ingredients", "diet", "flavor", "course", "state", "region", "style", "category", "source"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for dish in dishes:
            row = {
                "name": dish["name"],
                "ingredients": ", ".join(dish.get("ingredients") or []),
                "diet": dish.get("diet") or "",
                "flavor": dish.get("flavor") or "",
                "course": dish.get("course") or "",
                "state": dish.get("state") or "",
                "region": dish.get("region") or "",
                "style": dish.get("style") or "",
                "category": dish.get("category") or "",
                "source": dish.get("source") or "",
            }
            writer.writerow(row)
    print(f"  Saved {len(dishes)} dishes → {path}")


def print_stats(name: str, pairs: list[tuple[str, str]]):
    """Log pair count and unique query/dish counts."""
    unique_queries = len(set(q for q, _ in pairs))
    unique_dishes = len(set(d for _, d in pairs))
    print(f"  {name}: {len(pairs):,} pairs | {unique_queries:,} unique queries | {unique_dishes} unique dishes")


def main():
    print("Stage 1: Load & clean")
    dishes_old = load_indian_food(RAW_INDIAN_FOOD)
    print(f"  Loaded {len(dishes_old)} dishes from indian_food.csv")

    dishes_new = []
    if USE_ARCHANAS and RAW_ARCHANAS.exists():
        dishes_new = load_archanas_kitchen(RAW_ARCHANAS)
        print(f"  Loaded {len(dishes_new)} unique English dishes from IndianFoodDatasetCSV.csv")
    elif USE_ARCHANAS:
        print("  IndianFoodDatasetCSV.csv not found — skipping.")
    else:
        print("  Archana's Kitchen disabled (USE_ARCHANAS=False). Using Indian Food 101 only.")
    old_names = {d["name_lower"] for d in dishes_old}
    new_unique = [d for d in dishes_new if d["name_lower"] not in old_names]
    dishes = dishes_old + new_unique

    swiggy_bangalore_dishes = load_swiggy_bangalore_dishes(RAW_SWIGGY_BANGALORE)
    existing_names = {d["name_lower"] for d in dishes}
    swiggy_only = [d for d in swiggy_bangalore_dishes if d["name_lower"] not in existing_names]
    dishes = dishes + swiggy_only
    print(f"\n  Catalog: {len(dishes_old)} (Indian Food 101)"
          + (f" + {len(new_unique)} (Archana's unique)" if new_unique else "")
          + (f" + {len(swiggy_only)} (Swiggy Bangalore unique)" if swiggy_only else "")
          + f" = {len(dishes)} total dishes")

    swiggy_categories = extract_swiggy_food_categories(RAW_SWIGGY)
    bangalore_categories = extract_swiggy_bangalore_categories(RAW_SWIGGY_BANGALORE)
    swiggy_lower = {c.lower() for c in swiggy_categories}
    all_categories = list(swiggy_categories) + [c for c in bangalore_categories if c.lower() not in swiggy_lower]
    print(f"  Categories: {len(swiggy_categories)} from swiggy_cleaned, {len(bangalore_categories)} from Bangalore; {len(all_categories)} combined")

    ingredient_index = build_ingredient_index(dishes)
    print(f"  Built ingredient index: {len(ingredient_index)} unique ingredients")

    metadata_index = build_metadata_index(dishes)
    for key, vals in metadata_index.items():
        print(f"  Metadata[{key}]: {len(vals)} unique values")
    CHAAT_STYLE_NAMES = {
        "pani puri", "dahi vada", "sev khamani", "kachori", "samosa", "papad", "farsi puri",
        "vada", "dhokla", "aloo tikki", "papadum", "khaman", "lilva kachori", "ghooghra",
        "patra", "sev tameta", "chakali", "namakpara", "chevdo", "chorafali", "muthiya",
        "kutchi dabeli", "handwo", "khakhra", "khandvi", "masala dosa", "dosa", "uttapam",
    }
    for dish in dishes:
        dish.setdefault("style", "")
        dish.setdefault("category", None)
        dish.setdefault("source", "")
        dish["style"] = "chaat" if dish["name_lower"] in CHAAT_STYLE_NAMES else dish["style"]

    save_dish_catalog(dishes, PROCESSED_DIR / "dishes.csv")
    print("Stage 2: Programmatic pairs")

    exact_pairs = generate_exact_match_pairs(dishes)
    print_stats("Exact match", exact_pairs)

    partial_pairs = generate_partial_name_pairs(dishes)
    print_stats("Partial name", partial_pairs)

    ingredient_pairs = generate_ingredient_query_pairs(dishes, ingredient_index)
    print_stats("Ingredient", ingredient_pairs)

    category_pairs = generate_category_query_pairs(dishes, metadata_index, all_categories)
    print_stats("Category", category_pairs)

    cuisine_pairs = generate_cuisine_query_pairs(dishes)
    print_stats("Cuisine", cuisine_pairs)

    attribute_pairs = generate_attribute_query_pairs(dishes)
    print_stats("Attribute", attribute_pairs)

    occasion_pairs = generate_occasion_query_pairs(dishes)
    print_stats("Occasion", occasion_pairs)

    hinglish_pairs = generate_hinglish_query_pairs(dishes)
    print_stats("Hinglish (prog)", hinglish_pairs)

    synthetic_llm_pairs = generate_synthetic_llm_style_pairs(dishes)
    print_stats("Synthetic LLM-style", synthetic_llm_pairs)

    all_programmatic = (exact_pairs + partial_pairs + ingredient_pairs
                        + category_pairs + cuisine_pairs + attribute_pairs
                        + occasion_pairs + hinglish_pairs + synthetic_llm_pairs)
    print_stats("Total programmatic", all_programmatic)
    print("Stage 3: Augment")

    augmented = augment_pairs(all_programmatic)
    print_stats("Augmented (new)", augmented)

    all_pairs = all_programmatic + augmented
    print("Stage 4: LLM pairs")

    valid_dishes = {d["name_lower"] for d in dishes}
    llm_pairs = load_llm_pairs(LLM_DIR, valid_dishes=valid_dishes, dishes=dishes)
    if llm_pairs:
        print_stats("LLM-generated", llm_pairs)
        all_pairs = all_pairs + llm_pairs
    else:
        print("  No llm_queries*.json in data/raw/. Skipping.")
    print("Stage 5: Dedupe, split, save")

    all_pairs = deduplicate_pairs(all_pairs)
    print_stats("After dedup", all_pairs)

    dish_to_source = {d["name"]: d.get("source", "unknown") for d in dishes}
    train_pairs, val_pairs = split_train_val(
        all_pairs, val_fraction=0.1, dish_to_source=dish_to_source
    )
    print_stats("Train", train_pairs)
    print_stats("Val", val_pairs)

    save_pairs(train_pairs, PROCESSED_DIR / "train.csv")
    save_pairs(val_pairs, PROCESSED_DIR / "val.csv")
    print("Done. Outputs in data/processed/")


if __name__ == "__main__":
    main()
