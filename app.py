import sys
from PIL import Image
from multimodal_manager import MultimodalVectorEngine

def create_mock_image():
    """Generates an in-memory image object to simulate uploading a physical menu."""
    return Image.new('RGB', (300, 400), color=(230, 50, 50))

def main():
    engine = MultimodalVectorEngine()
    
    # Pre-seed the Multimodal Vector Index database with sample profiles
    # This seeds text profiles, metadata matrices, and simulated image uploads
    mock_menu_img = create_mock_image()
    engine.index_restaurant_profile(
        name="Pizza Palace", 
        cuisine="Italian", 
        menu_text="Authentic brick-oven thin pepperoni pizza, garlic knots, creamy tiramisu.",
        menu_image=mock_menu_img
    )
    engine.index_restaurant_profile(
        name="Taco Truck Express", 
        cuisine="Mexican", 
        menu_text="Spicy beef street tacos with cilantro, lime, corn chips and hot salsa salsa.",
        menu_image=mock_menu_img
    )
    engine.index_restaurant_profile(
        name="Luigi Trattoria", 
        cuisine="Italian", 
        menu_text="Homemade baked lasagna, penne carbonara, garlic bread slices, red wine.",
        menu_image=mock_menu_img
    )

    print("==================================================")
    print("🚀 MULTIMODAL VECTOR FUSION & RETRIEVAL ENGINE")
    print("==================================================")
    print("✅ Multimodal database loaded with text & image feature embeddings.")

    while True:
        print("\n[1] Cross-Modal Search (Text Query)")
        print("[2] Advanced Fusion Query (Text + Image Match)")
        print("[3] Exit Sandbox")
        choice = input("Select Operation: ").strip()

        if choice == "1":
            query = input("\nEnter semantic text query (e.g., 'pepperoni'): ").strip()
            cuisine_filter = input("Apply Metadata Filter? Enter Cuisine (or press Enter to skip): ").strip()
            
            meta_dict = {"cuisine_type": cuisine_filter} if cuisine_filter else None
            
            # Execute retrieval algorithm with metadata filtering constraint
            results = engine.retrieve_and_fuse(query_text=query, metadata_filter=meta_dict)
            
            print(f"\n🔍 RETRIEVAL RESULTS FOR: '{query}' (Filter: {cuisine_filter or 'None'})")
            print("-" * 65)
            if not results:
                print("❌ No matching profiles cleared the metadata filters.")
            for rank, res in enumerate(results, 1):
                print(f"Rank #{rank} | {res.restaurant_name:<18} | Score: {res.combined_score:<6} | Type: {res.match_type}")

        elif choice == "2":
            query = input("\nEnter text prompt part of search: ").strip()
            print("📷 Simulating parallel image capture from device upload...")
            uploaded_img = create_mock_image()
            
            weight_input = input("Set text fusion weight bias (0.0 to 1.0, e.g., 0.7): ").strip()
            t_weight = float(weight_input) if weight_input else 0.5
            
            # Execute weighted multimodal fusion and ranking calculations
            results = engine.retrieve_and_fuse(
                query_text=query, 
                query_image=uploaded_img, 
                text_weight=t_weight
            )
            
            print(f"\n⚡ FUSED MULTIMODAL RANKING MATRIX (Text Weight: {t_weight}, Image Weight: {round(1.0-t_weight, 2)})")
            print("-" * 65)
            for rank, res in enumerate(results, 1):
                print(f"Rank #{rank} | {res.restaurant_name:<18} | Fused Cosine Score: {res.combined_score:<6} | {res.match_type}")

        elif choice == "3":
            print("Powering down cross-modal matrix components. Goodbye!")
            break

if __name__ == "__main__":
    main()
