"""
One-time avatar generation script. Run this once, pick the best candidates,
set AVATAR_MALE_PATH and AVATAR_FEMALE_PATH in .env.

Saves all candidates to GCS (DEV_MODE=false) or local_storage/shared/avatars/ (DEV_MODE=true).

Usage:
  cd backend
  uv run python scripts/generate_avatars.py

After running:
  1. Review candidates in GCS or local_storage/shared/avatars/
  2. Pick the best of each gender
  3. Set in .env:
       AVATAR_MALE_PATH=gs://nevertrtfm/shared/avatars/avatar_male.jpg
       AVATAR_FEMALE_PATH=gs://nevertrtfm/shared/avatars/avatar_female.jpg
"""

from dotenv import load_dotenv
load_dotenv()

from google.genai import types
from tools.gemini import build_client
from tools.storage import save_shared, DEV_MODE

AVATARS = {
    "male": """
        A professional South Asian male presenter in his late 20s, wearing a
        navy blue blazer over a white t-shirt, short groomed beard, stylish
        glasses. Looking directly at camera with a confident neutral expression.
        Soft studio three-point lighting, shallow depth of field, neutral
        high-tech office background with bokeh. Portrait orientation,
        photorealistic, 4K, professional headshot quality.
    """,
    "female": """
        A professional East Asian female presenter in her late 20s, wearing a
        tailored charcoal blazer over a light blouse, minimal jewelry, hair
        pulled back neatly. Looking directly at camera with a calm authoritative
        expression. Soft studio three-point lighting, shallow depth of field,
        neutral high-tech office background with bokeh. Portrait orientation,
        photorealistic, 4K, professional headshot quality.
    """,
}


def main():
    client = build_client()
    mode = "local" if DEV_MODE else "GCS"
    print(f"Saving avatars to {mode}...\n")

    saved = {}
    for gender, prompt in AVATARS.items():
        print(f"Generating {gender} avatar candidates...")
        response = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt.strip(),
            config=types.GenerateImagesConfig(
                number_of_images=4,
                aspect_ratio="9:16",
                safety_filter_level="block_some",
                person_generation="allow_adult",
            ),
        )

        for i, image in enumerate(response.generated_images):
            uri = save_shared(f"avatars/avatar_{gender}_candidate_{i}.jpg", image.image.image_bytes)
            print(f"  Saved candidate {i}: {uri}")

        # Save candidate 0 as the default — replace manually after reviewing
        default_uri = save_shared(f"avatars/avatar_{gender}.jpg", response.generated_images[0].image.image_bytes)
        saved[gender] = default_uri
        print(f"  Default → {default_uri}\n")

    print("Done. Review candidates and update .env with the best paths:")
    print(f"  AVATAR_MALE_PATH={saved['male']}")
    print(f"  AVATAR_FEMALE_PATH={saved['female']}")
    print()
    print("To swap to a different candidate, copy it:")
    if DEV_MODE:
        print("  cp local_storage/shared/avatars/avatar_male_candidate_X.jpg local_storage/shared/avatars/avatar_male.jpg")
    else:
        print("  gcloud storage cp gs://nevertrtfm/shared/avatars/avatar_male_candidate_X.jpg gs://nevertrtfm/shared/avatars/avatar_male.jpg")


if __name__ == "__main__":
    main()
