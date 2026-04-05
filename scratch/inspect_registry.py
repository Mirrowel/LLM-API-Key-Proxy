import asyncio
import logging
from rotator_library.model_info_service import get_model_info_service

async def main():
    service = get_model_info_service()
    # Ensure it's started and has data
    await service.start()
    await asyncio.sleep(2) # Wait for initial fetch
    
    models = service.get_all_source_models()
    print(f"Total models in registry: {len(models)}")
    
    google_models = [mid for mid in models if mid.startswith("google/") or "/google/" in mid]
    print(f"Google models found: {len(google_models)}")
    for mid in google_models[:10]:
        print(f"  {mid}")

if __name__ == "__main__":
    asyncio.run(main())
