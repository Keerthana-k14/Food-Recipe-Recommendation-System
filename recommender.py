import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import numpy as np
import urllib.parse

print("🚀 Initializing Food Recommendation System with Detailed Recipe Steps...")

# Load data
try:
    recipes = pd.read_csv('data/RAW_recipes.csv')
    print(f"✅ Loaded {len(recipes)} recipes")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    # Create sample data
    recipes = pd.DataFrame({
        'name': [
            'Tomato Pasta', 'Vegetable Fried Rice', 'Tomato Salad', 'Onion Soup',
            'Chicken Curry', 'Egg Fried Rice', 'Vegetable Soup', 'Chicken Rice',
            'Tomato Egg Curry', 'Mixed Vegetable Pasta', 'Onion Salad', 'Rice Pudding',
            'Masala Dosa', 'Biryani', 'Butter Chicken', 'Paneer Tikka',
            'Borracho Chicken', 'Drunken Beer Can Chicken', 'Spicy Chicken Wings',
            'Chicken Alfredo Pasta', 'Chicken Biryani', 'Butter Chicken Masala'
        ],
        'description': [
            'Simple pasta with fresh tomatoes', 'Fried rice with vegetables',
            'Fresh salad with tomatoes', 'Classic French onion soup',
            'Spicy chicken curry', 'Quick fried rice with eggs',
            'Hearty vegetable soup', 'Flavorful chicken rice',
            'Tomato and egg curry', 'Pasta with mixed vegetables',
            'Simple onion salad', 'Sweet rice dessert',
            'Crispy South Indian dosa', 'Fragrant rice dish with spices',
            'Creamy chicken in tomato sauce', 'Grilled cottage cheese',
            'Mexican drunken chicken with beer', 'Beer can chicken recipe',
            'Hot and spicy chicken wings', 'Creamy chicken pasta',
            'Fragrant chicken rice dish', 'Creamy Indian chicken curry'
        ],
        'ingredients': [
            'tomato, pasta, garlic, oil', 'rice, vegetables, egg, soy sauce',
            'tomato, onion, cucumber', 'onion, broth, bread, cheese',
            'chicken, onion, tomato, spices', 'rice, egg, vegetables',
            'vegetables, broth, herbs', 'chicken, rice, spices',
            'tomato, egg, spices', 'pasta, vegetables, tomato',
            'onion, cucumber, lemon', 'rice, milk, sugar',
            'rice, lentils, spices', 'rice, meat, spices, herbs',
            'chicken, cream, tomato, spices', 'paneer, spices, yogurt',
            'chicken, beer, spices, lime', 'chicken, beer, herbs',
            'chicken wings, spices, sauce', 'chicken, pasta, cream, cheese',
            'chicken, rice, spices, herbs', 'chicken, butter, cream, spices'
        ],
        'rating': [4.5, 4.3, 4.0, 4.6, 4.7, 4.2, 4.1, 4.4, 4.3, 4.2, 3.9, 4.5, 4.8, 4.9, 4.7, 4.6, 4.5, 4.4, 4.3, 4.6,
                   4.8, 4.7]
    })

# Ensure required columns and clean data
if 'rating' not in recipes.columns:
    recipes['rating'] = 4.0

if 'ingredients' not in recipes.columns:
    recipes['ingredients'] = 'Various ingredients'

# Clean the name column - convert all to strings and handle NaN
recipes['name'] = recipes['name'].fillna('Unknown Recipe').astype(str)
recipes['description'] = recipes['description'].fillna('No description').astype(str)
recipes['ingredients'] = recipes['ingredients'].fillna('No ingredients').astype(str)


def generate_detailed_recipe_steps(recipe_name, ingredients):
    """Generate detailed 10+ step cooking instructions for each recipe"""
    recipe_name = str(recipe_name).lower()
    ingredients = str(ingredients).lower()

    # Recipe-specific detailed instructions
    recipe_steps_db = {
        'borracho chicken': """1. PREPARE THE CHICKEN: Clean and pat dry a whole chicken (3-4 lbs)
2. MAKE THE MARINADE: In a bowl, mix 1 cup beer, 3 tbsp lime juice, 2 tbsp olive oil, 4 minced garlic cloves, 1 tbsp chili powder, 1 tsp cumin, 1 tsp oregano, salt and pepper
3. MARINATE: Rub the marinade all over the chicken, including under the skin. Cover and refrigerate for at least 4 hours or overnight
4. PREPARE BEER CAN: Take a 12 oz beer can and drink or pour out half of the beer
5. ADD FLAVORS: Add 2 crushed garlic cloves, 1 tsp chili powder, and a spring of rosemary to the beer can
6. SET UP GRILL: Preheat grill to medium-high heat (375°F). Set up for indirect grilling
7. POSITION CHICKEN: Place the beer can on a stable surface. Carefully lower the chicken cavity onto the beer can so it stands upright
8. GRILL: Place the chicken on the grill in the indirect heat zone. Cover and cook for 60-75 minutes
9. CHECK TEMPERATURE: Insert meat thermometer into thigh - should read 165°F
10. REST: Carefully remove chicken from grill using tongs. Let rest for 15 minutes before carving
11. MAKE SAUCE: While chicken rests, mix remaining marinade with 2 tbsp honey and simmer for 5 minutes
12. SERVE: Carve chicken and serve with the sauce, grilled vegetables, and tortillas""",

        'chicken curry': """1. PREPARE INGREDIENTS: Cut 2 lbs chicken into cubes. Chop 2 onions, 3 tomatoes, and 4 garlic cloves
2. MARINATE CHICKEN: Mix chicken with 2 tbsp yogurt, 1 tsp turmeric, 1 tsp salt. Marinate 30 minutes
3. HEAT OIL: Heat 3 tbsp oil in a large pot over medium heat
4. SAUTE AROMATICS: Add 1 tsp cumin seeds, then add chopped onions. Cook until golden brown
5. ADD GARLIC & GINGER: Add 1 tbsp grated ginger and 4 minced garlic cloves. Cook 2 minutes
6. ADD SPICES: Add 2 tbsp curry powder, 1 tsp coriander powder, 1/2 tsp chili powder. Cook 1 minute
7. ADD TOMATOES: Add chopped tomatoes and cook until soft and oil separates (8-10 minutes)
8. COOK CHICKEN: Add marinated chicken and cook until sealed (5-7 minutes)
9. ADD LIQUID: Add 1 cup coconut milk and 1/2 cup water. Bring to simmer
10. SIMMER: Cover and simmer on low heat for 25-30 minutes until chicken is tender
11. FINISHING: Add 1/2 cup cream and 1 tbsp lemon juice. Stir gently
12. GARNISH: Garnish with fresh cilantro and serve with rice or naan""",

        'tomato pasta': """1. BOIL WATER: Bring 4 quarts of water to a rolling boil in a large pot
2. SALT WATER: Add 2 tbsp salt to the boiling water
3. COOK PASTA: Add 1 lb pasta and cook according to package directions until al dente
4. PREPARE SAUCE: While pasta cooks, heat 3 tbsp olive oil in a large pan
5. SAUTE GARLIC: Add 4 minced garlic cloves and cook until fragrant (30 seconds)
6. ADD TOMATOES: Add 4 cups chopped fresh tomatoes or 2 cans crushed tomatoes
7. SEASON: Add 1 tsp salt, 1/2 tsp black pepper, 1 tsp sugar, and 1/2 tsp red chili flakes
8. SIMMER SAUCE: Cook sauce for 15-20 minutes until thickened, stirring occasionally
9. ADD HERBS: Stir in 1/4 cup chopped fresh basil and 2 tbsp chopped parsley
10. RESERVE PASTA WATER: Before draining pasta, reserve 1 cup of pasta cooking water
11. COMBINE: Drain pasta and add to the sauce along with 1/2 cup pasta water
12. FINISH: Toss pasta with sauce, adding 2 tbsp butter and 1/2 cup grated Parmesan cheese
13. SERVE: Garnish with more Parmesan and fresh basil leaves""",

        'vegetable fried rice': """1. PREP RICE: Use 3 cups cold cooked rice (day-old works best)
2. CHOP VEGETABLES: Dice 1 carrot, 1/2 cup peas, 1 bell pepper, 2 green onions, 2 garlic cloves
3. BEAT EGGS: Beat 2 eggs with 1/4 tsp salt and pinch of pepper
4. HEAT WOK: Heat 2 tbsp oil in a wok or large pan over high heat
5. SCRAMBLE EGGS: Add eggs and scramble quickly. Remove and set aside
6. SAUTE VEGETABLES: Add 1 more tbsp oil, then add carrots and bell pepper. Stir-fry 2 minutes
7. ADD AROMATICS: Add minced garlic and white parts of green onions. Cook 1 minute
8. ADD RICE: Add cold rice, breaking up any clumps. Stir-fry 3-4 minutes
9. SEASON: Add 3 tbsp soy sauce, 1 tsp sesame oil, 1/2 tsp white pepper
10. COMBINE: Return eggs to wok and add peas. Mix everything well
11. FINISH: Add green onion tops and 1/2 cup corn if desired
12. SERVE: Garnish with extra green onions and serve hot""",

        'butter chicken': """1. MARINATE CHICKEN: Mix 2 lbs chicken with 1 cup yogurt, 2 tbsp lemon juice, 2 tbsp ginger-garlic paste, 1 tsp turmeric, 2 tsp chili powder, salt. Marinate 4 hours
2. PREPARE SAUCE: Soak 10-12 cashews in warm water for 30 minutes
3. COOK CHICKEN: Grill or bake marinated chicken until cooked. Set aside
4. MAKE TOMATO GRAVY: Heat 2 tbsp butter, add 2 chopped onions. Cook until golden
5. ADD TOMATOES: Add 4 chopped tomatoes, cook until soft
6. BLEND SAUCE: Cool tomato mixture, blend with soaked cashews to smooth paste
7. COOK SAUCE: Heat 3 tbsp butter, add 1 tbsp ginger-garlic paste. Cook 1 minute
8. ADD SPICES: Add 2 tsp garam masala, 1 tsp cumin powder, 1 tsp coriander powder
9. COMBINE: Add blended tomato-cashew paste. Cook 10 minutes
10. ADD CREAM: Add 1 cup heavy cream and 1/2 cup milk. Simmer 5 minutes
11. ADD CHICKEN: Add cooked chicken pieces. Simmer 10 minutes
12. FINISH: Add 1 tbsp kasuri methi, 1 tbsp honey, 2 tbsp butter. Garnish with cream""",

        'biryani': """1. MARINATE MEAT: Mix 2 lbs meat with yogurt, spices, and lemon juice. Marinate 2 hours
2. SOAK RICE: Soak 3 cups basmati rice for 30 minutes
3. PREPARE LAYERS: Slice 2 onions thinly, chop 2 tomatoes, make biryani masala
4. FRY ONIONS: Deep fry sliced onions until golden brown for birista
5. COOK MEAT: Cook marinated meat with whole spices until 70% done
6. PARBOIL RICE: Boil soaked rice with whole spices until 70% cooked
7. LAYER BIRYANI: In heavy pot, layer rice, meat, fried onions, mint, cilantro
8. ADD COLOR: Add saffron milk and food color in patterns
9. SEAL POT: Cover with lid and seal with dough or aluminum foil
10. DUUM COOKING: Cook on very low heat for 25-30 minutes
11. REST: Let biryani rest for 15 minutes before opening
12. SERVE: Fluff gently and serve with raita and salad""",

        'masala dosa': """1. SOAK RICE & LENTILS: Soak 2 cups rice and 1/2 cup urad dal separately for 6 hours
2. GRIND BATTER: Grind rice and dal separately, then mix together
3. FERMENT: Add salt, cover, and ferment batter for 8-12 hours
4. PREPARE POTATO FILLING: Boil 4 potatoes until tender. Peel and mash roughly
5. TEMPER SPICES: Heat oil, add mustard seeds, urad dal, chana dal, curry leaves
6. MAKE BHUNI: Add chopped onions, green chilies, ginger. Cook until soft
7. ADD POTATOES: Add mashed potatoes, turmeric, salt. Mix well
8. HEAT DOSA PAN: Heat a cast iron or non-stick tawa over medium heat
9. SPREAD DOSA: Pour ladle of batter, spread in circular motion to make thin dosa
10. COOK DOSA: Drizzle oil around edges. Cook until golden and crisp
11. ADD FILLING: Place potato filling in center, fold dosa over
12. SERVE: Serve hot with coconut chutney and sambar"""
    }

    # Find matching recipe or generate generic steps
    for key, steps in recipe_steps_db.items():
        if key in recipe_name:
            return steps

    # Generic detailed steps for any recipe
    return f"""1. PREPARE INGREDIENTS: Gather and measure all ingredients listed. Wash and chop vegetables as needed
2. PREPARE PROTEIN: Clean, trim, and cut protein into appropriate sizes. Pat dry with paper towels
3. MAKE MARINADE: Combine spices, oils, acids, and seasonings in a bowl for marinating
4. MARINATE: Coat protein evenly with marinade. Cover and refrigerate for recommended time
5. PREHEAT COOKING SURFACE: Preheat oven, grill, or stovetop to required temperature
6. PREPARE COOKING VESSEL: Grease pans, line baking sheets, or prepare cooking surface
7. COOK AROMATICS: Sauté onions, garlic, ginger in oil until fragrant and translucent
8. ADD SPICES: Toast whole spices or add ground spices to release flavors
9. COOK MAIN INGREDIENTS: Add main components and cook until partially done
10. ADD LIQUIDS: Pour in stocks, sauces, or water. Bring to simmer
11. SIMMER: Cover and cook on low heat until all ingredients are tender and flavors meld
12. FINISHING TOUCHES: Add fresh herbs, cream, lemon juice, or final seasonings
13. CHECK SEASONING: Taste and adjust salt, pepper, and spices as needed
14. REST: Let the dish rest for proper texture and flavor development
15. GARNISH & SERVE: Add final garnishes and serve hot with appropriate sides
16. STORE LEFTOVERS: Cool completely and store in airtight containers"""


# Add detailed recipe steps to all recipes
recipes['recipe_steps'] = recipes.apply(
    lambda row: generate_detailed_recipe_steps(row['name'], row['ingredients']),
    axis=1
)


def get_exact_recipe_youtube_link(recipe_name):
    """Get EXACT YouTube search link for the SPECIFIC RECIPE NAME"""
    recipe_name = str(recipe_name).strip()

    if not recipe_name or recipe_name.lower() in ['nan', 'unknown recipe', '']:
        return 'https://www.youtube.com/results?search_query=cooking+recipes'

    exact_search = f"{recipe_name} recipe"
    encoded_query = urllib.parse.quote_plus(exact_search)
    youtube_url = f"https://www.youtube.com/results?search_query={encoded_query}"

    print(f"🎬 YouTube searching for EXACT recipe: '{exact_search}'")
    return youtube_url


# Add YouTube links to recipes
recipes['youtube_link'] = recipes['name'].apply(get_exact_recipe_youtube_link)

# Prepare features for search
recipes['combined_features'] = (
        recipes['name'] + ' ' +
        recipes['description'] + ' ' +
        recipes['ingredients']
)

# Train model
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = vectorizer.fit_transform(recipes['combined_features'])

print("✅ Model trained successfully")
print("📝 Added detailed recipe steps with 10+ instructions for each recipe!")


def find_recipes_by_ingredients(user_ingredients, top_n=6):
    """Find recipes based on ingredients"""
    if not user_ingredients or not user_ingredients.strip():
        return []

    user_ingredients = user_ingredients.lower().strip()
    user_ingredient_list = [ing.strip() for ing in user_ingredients.split(',') if ing.strip()]

    if not user_ingredient_list:
        return []

    matches = []

    for idx, row in recipes.iterrows():
        recipe_ingredients = str(row.get('ingredients', '')).lower()

        # Count matching ingredients
        match_count = sum(1 for user_ing in user_ingredient_list if user_ing in recipe_ingredients)

        if match_count > 0:
            match_percentage = (match_count / len(user_ingredient_list)) * 100

            matches.append({
                'name': row['name'],
                'description': row['description'],
                'ingredients': row['ingredients'],
                'recipe_steps': row['recipe_steps'],  # ADDED DETAILED STEPS
                'youtube_link': row['youtube_link'],
                'match_percentage': match_percentage,
                'rating': row.get('rating', 4.0)
            })

    matches.sort(key=lambda x: (x['match_percentage'], x['rating']), reverse=True)
    return matches[:top_n]


def recommend_food(search_query, top_n=5):
    """Search recipes by name"""
    try:
        query_vec = vectorizer.transform([search_query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

        if 'rating' in recipes.columns:
            ratings = recipes['rating'].fillna(0).values
            similarities = similarities * (ratings / 5.0)

        top_indices = similarities.argsort()[-top_n:][::-1]
        results = recipes.iloc[top_indices].copy()

        return results
    except Exception as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()


def get_food_image(food_name):
    """Get food image from Unsplash"""
    food_name = str(food_name).strip()
    if not food_name or food_name.lower() in ['nan', 'unknown recipe']:
        food_name = "food"

    food_clean = food_name.replace(' ', '+')
    return f"https://source.unsplash.com/400x300/?{food_clean},food"


# Helper function to count steps safely
def count_steps_safely(steps_text):
    """Safely count the number of steps in recipe instructions"""
    if not steps_text:
        return 0
    steps_list = steps_text.split('\n')
    return len([step for step in steps_list if step.strip() and step.strip()[0].isdigit()])


# Test the system
if __name__ == "__main__":
    print("🧪 Testing detailed recipe steps...")

    test_recipes = ["Borracho Chicken", "Chicken Curry", "Tomato Pasta"]
    for recipe in test_recipes:
        steps = generate_detailed_recipe_steps(recipe, "")
        step_count = count_steps_safely(steps)
        print(f"📋 {recipe} steps: {step_count} steps")
        first_steps = steps.split('\n')[:3]
        print(f"First 3 steps: {first_steps}")

    print("✅ Detailed recipe steps added successfully!")