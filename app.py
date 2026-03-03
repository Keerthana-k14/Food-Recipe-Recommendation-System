import streamlit as st
import pandas as pd
from recommender import recommend_food, find_recipes_by_ingredients, get_food_image
import time

# Page configuration
st.set_page_config(
    page_title="🍽️ Food Recipe Finder",
    page_icon="🍽️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .food-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .food-name {
        color: #2E86AB;
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .youtube-btn {
        background-color: #FF0000 !important;
        color: white !important;
        border: none !important;
        padding: 12px 20px !important;
        border-radius: 25px !important;
        font-weight: bold !important;
        margin: 10px 0 !important;
        text-decoration: none !important;
        display: inline-block !important;
        width: 100%;
        text-align: center;
        font-size: 1rem;
    }
    .youtube-btn:hover {
        background-color: #CC0000 !important;
        transform: scale(1.05);
    }
    .match-badge {
        background: linear-gradient(45deg, #FF6B6B, #FF8E53);
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .rating-stars {
        color: #FFD700;
        font-size: 1.2rem;
        margin: 8px 0;
    }
    .search-info {
        background: #e8f4fd;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        font-size: 0.9rem;
    }
    .recipe-steps {
        background: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
        margin: 10px 0;
        white-space: pre-line;
        font-family: 'Courier New', monospace;
        line-height: 1.6;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">🍽️Recipe Recommendation System</h1>', unsafe_allow_html=True)

# Initialize session state
if 'last_results' not in st.session_state:
    st.session_state.last_results = None
if 'ingredient_results' not in st.session_state:
    st.session_state.ingredient_results = None

# Tabs for search types
tab1, tab2 = st.tabs(["🔍 Search by Name", "🥗 Search by Ingredients"])

with tab1:
    st.markdown("### Find Recipes by Name")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        search_query = st.text_input(
            "Enter food name",
            placeholder="e.g., pasta, biryani, curry...",
            value="chicken",
            key="name_search"
        )

    with col2:
        top_n = st.selectbox("Results", [3, 5, 8], index=1, key="name_count")

    with col3:
        st.write("")
        st.write("")
        if st.button("🚀 Search Recipes", key="name_btn", use_container_width=True):
            if search_query.strip():
                with st.spinner('Searching for exact recipes...'):
                    results = recommend_food(search_query, top_n)
                    st.session_state.last_results = results
                    st.session_state.ingredient_results = None

with tab2:
    st.markdown("### Find Recipes by Ingredients")

    col1, col2 = st.columns([3, 1])

    with col1:
        user_ingredients = st.text_input(
            "Enter ingredients (comma separated)",
            placeholder="tomato, onion, rice, chicken...",
            key="ingredient_input"
        )

    with col2:
        st.write("")
        st.write("")
        if st.button("🔍 Find Recipes", key="ingredient_btn", use_container_width=True):
            if user_ingredients.strip():
                with st.spinner('Finding exact recipe matches...'):
                    ingredient_results = find_recipes_by_ingredients(user_ingredients, 6)
                    st.session_state.ingredient_results = ingredient_results
                    st.session_state.last_results = None


# Helper function to count steps safely
def count_recipe_steps(recipe_steps):
    """Safely count the number of steps in recipe instructions"""
    if not recipe_steps:
        return 0
    steps_list = recipe_steps.split('\n')
    return len([step for step in steps_list if step.strip() and step.strip()[0].isdigit()])


# Display name search results
if st.session_state.last_results is not None and not st.session_state.last_results.empty:
    results = st.session_state.last_results
    st.success(f"🎉 Found {len(results)} exact recipes for '{search_query}'!")

    cols = st.columns(2)

    for idx, (_, row) in enumerate(results.iterrows()):
        with cols[idx % 2]:
            with st.container():
                st.markdown('<div class="food-card">', unsafe_allow_html=True)

                # Food name
                st.markdown(f'<div class="food-name">{row["name"]}</div>', unsafe_allow_html=True)

                # Food image
                try:
                    image_url = get_food_image(row['name'])
                    st.image(image_url, use_container_width=True, caption=f"🍽️ {row['name']}")
                except:
                    st.info(f"📷 Image for: {row['name']}")

                # Description
                desc = row.get('description', '')
                if desc and str(desc).strip().lower() not in ['nan', 'none', 'no description']:
                    st.write(f"📝 {desc}")

                # Ingredients
                ingredients = row.get('ingredients', '')
                if ingredients and str(ingredients).strip().lower() not in ['nan', 'none', 'no ingredients']:
                    with st.expander("📋 Ingredients"):
                        st.write(ingredients)

                # RECIPE STEPS
                recipe_steps = row.get('recipe_steps', '')
                if recipe_steps:
                    step_count = count_recipe_steps(recipe_steps)
                    with st.expander(f"👩‍🍳 Detailed Cooking Steps ({step_count} Steps)"):
                        st.markdown(f'<div class="recipe-steps">{recipe_steps}</div>', unsafe_allow_html=True)
                        st.info(f"📋 Total steps: {step_count} detailed instructions")

                # Show what YouTube will search for
                st.markdown(f'<div class="search-info">🔍 YouTube will search for: <b>"{row["name"]} recipe"</b></div>',
                            unsafe_allow_html=True)

                # YouTube button
                youtube_link = row['youtube_link']
                st.markdown(f'''
                <a href="{youtube_link}" target="_blank">
                    <button class="youtube-btn">
                        🎬 Watch "{row['name']}" Tutorial
                    </button>
                </a>
                ''', unsafe_allow_html=True)

                # Rating
                rating = row.get('rating', 4.0)
                try:
                    rating_num = float(rating)
                    full_stars = int(rating_num)
                    empty_stars = 5 - full_stars
                    rating_display = "⭐" * full_stars + "☆" * empty_stars
                    rating_value = f"{rating_num:.1f}"
                    st.markdown(f'<div class="rating-stars">{rating_display} ({rating_value}/5)</div>',
                                unsafe_allow_html=True)
                except (ValueError, TypeError):
                    st.write(f"⭐ Rating: {rating}/5")

                st.markdown('</div>', unsafe_allow_html=True)

# Display ingredient search results
elif st.session_state.ingredient_results is not None:
    results = st.session_state.ingredient_results

    if results:
        st.success(f"🎉 Found {len(results)} exact matching recipes!")

        cols = st.columns(2)

        for idx, recipe in enumerate(results):
            with cols[idx % 2]:
                with st.container():
                    st.markdown('<div class="food-card">', unsafe_allow_html=True)

                    # Food name and match percentage
                    st.markdown(f'<div class="food-name">{recipe["name"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="match-badge">🎯 {recipe["match_percentage"]:.0f}% Match</div>',
                                unsafe_allow_html=True)

                    # Food image
                    try:
                        image_url = get_food_image(recipe['name'])
                        st.image(image_url, use_container_width=True, caption=f"🍽️ {recipe['name']}")
                    except:
                        st.info(f"📷 Image for: {recipe['name']}")

                    # Description
                    if recipe['description'] and recipe['description'] != 'No description':
                        st.write(f"📝 {recipe['description']}")

                    # Ingredients
                    with st.expander("📋 Ingredients"):
                        st.write(recipe['ingredients'])

                    # RECIPE STEPS
                    if recipe.get('recipe_steps'):
                        step_count = count_recipe_steps(recipe['recipe_steps'])
                        with st.expander(f"👩‍🍳 Detailed Cooking Steps ({step_count} Steps)"):
                            st.markdown(f'<div class="recipe-steps">{recipe["recipe_steps"]}</div>',
                                        unsafe_allow_html=True)
                            st.info(f"📋 Total steps: {step_count} detailed instructions")

                    # Show what YouTube will search for
                    st.markdown(
                        f'<div class="search-info">🔍 YouTube will search for: <b>"{recipe["name"]} recipe"</b></div>',
                        unsafe_allow_html=True)

                    # YouTube button
                    st.markdown(f'''
                    <a href="{recipe['youtube_link']}" target="_blank">
                        <button class="youtube-btn">
                            🎬 Watch "{recipe['name']}" Tutorial
                        </button>
                    </a>
                    ''', unsafe_allow_html=True)

                    # Rating
                    rating = recipe['rating']
                    try:
                        rating_num = float(rating)
                        full_stars = int(rating_num)
                        empty_stars = 5 - full_stars
                        rating_display = "⭐" * full_stars + "☆" * empty_stars
                        rating_value = f"{rating_num:.1f}"
                        st.markdown(f'<div class="rating-stars">{rating_display} ({rating_value}/5)</div>',
                                    unsafe_allow_html=True)
                    except (ValueError, TypeError):
                        st.write(f"⭐ Rating: {rating}/5")

                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("😔 No exact recipes found. Try different ingredients!")

# Sidebar with quick searches
with st.sidebar:
    st.markdown("## ⚡ Quick Recipes")

    quick_recipes = [
        ("Chicken Recipes", "🍗"), ("Pasta", "🍝"), ("Biryani", "🍛"),
        ("Curry", "🥘"), ("Fried Rice", "🍚"), ("Salad", "🥗")
    ]

    for recipe, emoji in quick_recipes:
        if st.button(f"{emoji} {recipe}", key=f"quick_{recipe}", use_container_width=True):
            st.session_state.name_search = recipe.lower()
            st.rerun()

    st.markdown("## 🥗 Common Ingredients")

    ingredient_sets = [
        "chicken, spices, onion",
        "tomato, onion, rice",
        "eggs, vegetables, rice",
        "pasta, cheese, tomato"
    ]

    for ingredients in ingredient_sets:
        if st.button(f"🥗 {ingredients}", key=f"ing_{ingredients}", use_container_width=True):
            st.session_state.ingredient_input = ingredients
            st.rerun()

# Welcome message
if st.session_state.last_results is None and st.session_state.ingredient_results is None:
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👋 Welcome!")
        st.markdown("""
        **Find perfect recipes with DETAILED cooking steps!**

        🎯 **Search by Name:**
        - Find specific dishes
        - Get EXACT YouTube cooking links
        - Detailed 10+ step instructions

        🥗 **Search by Ingredients:**
        - Use what you have in kitchen
        - See match percentages
        - Get complete recipe guides
        """)

    with col2:
        st.markdown("### 🚀 How It Works")
        st.markdown("""
        **For Each Recipe You Get:**

        📖 **Ingredients List**
        - All required ingredients
        - Exact measurements

        👩‍🍳 **Detailed Cooking Steps**
        - 10-16 step instructions
        - Cooking times and temperatures
        - Professional tips

        🎬 **Exact YouTube Tutorials**
        - Specific recipe searches
        - Video demonstrations
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "🍽️ Food Recipe Finder | Detailed Cooking Steps | Professional Recipes 🍽️"
    "</div>",
    unsafe_allow_html=True
)