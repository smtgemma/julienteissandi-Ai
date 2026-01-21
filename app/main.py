from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import openai
import os
from dotenv import load_dotenv
import json
import traceback

# Load environment variables
load_dotenv()

# Initialize OpenAI client
try:
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1"  # Explicitly set the base URL
    )
except Exception as e:
    print(f"Error initializing OpenAI client: {str(e)}")
    raise

app = FastAPI(
    title="Tour Planning API",
    description="Generate day-by-day tour plans based on travel preferences",
    version="1.0.0"
)

# Pydantic models
class TourRequest(BaseModel):
    from_place: str = Field(..., description="Starting location")
    to_place: str = Field(..., description="Destination")
    duration: int = Field(..., ge=1, le=30, description="Tour duration in days")
    budget: str = Field(..., description="Budget level (e.g., 1000, 2000-3000, budget, luxury)")
    type: str = Field(..., description="Tour type (budget/luxury)")
    travelers: int = Field(..., ge=1, le=20, description="Number of travelers")


class TourPlanResponse(BaseModel):
    tour_plan: Dict[str, Any]

def generate_tour_plan(tour_requests: List[TourRequest]) -> Dict[str, Any]:
    """
    Generate tour plan using OpenAI API
    """
    try:
        # Create prompt for OpenAI
        prompt = create_prompt(tour_requests)
        
        print(f"Sending request to OpenAI with prompt length: {len(prompt)}")
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano",  # Fixed model name - use "gpt-4" if you have access
            messages=[
                {
                    "role": "system",
                    "content": """You are a professional travel planner. Create detailed day-by-day tour plans based on the user's requirements. 
                    Always respond with valid JSON format. Include specific activities, accommodations, and budget breakdowns. Also include the name of the place to visit and mention their co-ordinate in (). Make sure that you make the co-ordinates correct. This thing is crucial. 
                    Your response must be pure JSON without any additional text."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=22000,
            response_format={"type": "json_object"}  # Force JSON response
        )
        
        # Parse the response
        plan_content = response.choices[0].message.content.strip()
        print(f"Received response from OpenAI: {plan_content[:200]}...")
        
        # Extract JSON from response
        try:
            # Clean the response - remove any markdown code blocks
            cleaned_content = plan_content
            if "```json" in cleaned_content:
                cleaned_content = cleaned_content.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_content:
                cleaned_content = cleaned_content.split("```")[1].split("```")[0].strip()
            
            tour_plan = json.loads(cleaned_content)
            print("Successfully parsed JSON response")
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Raw content: {plan_content}")
    
        return tour_plan
        
    except Exception as e:
        print(f"Error in generate_tour_plan: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating tour plan: {str(e)}")

def create_prompt(tour_requests: List[TourRequest]) -> str:
    """
    Create a detailed prompt for OpenAI based on tour requests
    """
    # Build destination-specific text (do not overwrite this later)
    destinations_text = ""
    if len(tour_requests) > 1:
        destinations_text += "Please create a detailed multi-destination tour plan with the following requirements:\n\n"
    else:
        destinations_text += "Please create a detailed tour plan with the following requirements:\n\n"
    
    for i, request in enumerate(tour_requests, 1):
        destinations_text += f"Destination {i}:\n"
        destinations_text += f"- Starting from: {request.from_place}\n"
        destinations_text += f"- Destination: {request.to_place}\n"
        destinations_text += f"- Duration: {request.duration} days\n"
        destinations_text += f"- Budget: {request.budget}\n"
        destinations_text += f"- Type: {request.type}\n"
        destinations_text += f"- Number of travelers: {request.travelers}\n\n"
    
    total_days = sum(req.duration for req in tour_requests)
    
    # Include the previously built destinations_text into the final prompt
    prompt = f"""
{destinations_text}
Trip Overview:
- Total trip duration: {total_days} days
- Number of travelers: {tour_requests[0].travelers}
- Traveler's focus: Create a balanced, engaging experience — include sightseeing, local culture, relaxation, and food.
- Ensure the plan matches the given duration and budget level for {tour_requests[0].travelers} {'person' if tour_requests[0].travelers == 1 else 'people'}.
Please return the response strictly in **valid JSON format**, using the following structure:
{{
  "total_days": {total_days},
  "estimated_total_cost": "Overall estimated cost based on input budget(s)",
  "tour_plan": {{
    "day 1": {{
      "Plan": "Write a comprehensive and engaging paragraph describing the full itinerary for the day. Include:
        Morning, afternoon, and evening activities with estimated timings.
        Travel routes between locations, mentioning transport options (car, train, walking, etc.) and approximate travel time.
        Must-see attractions and hidden gems worth visiting, with a short reason why they're special.
        Meal suggestions for breakfast, lunch, and dinner, including local or recommended restaurants and their approximate costs.
        Any entry tickets, activity fees, or local transport costs.
        Suggestions for breaks, relaxation spots, or photo opportunities.
        Write in a clear, travel-guide tone, focusing on smooth flow and practical details for a real traveler.",
      "locations": [
        {{
          "name": "Location/Attraction name",
          "latitude": 0.0,
          "longitude": 0.0
        }}
      ]
    }},
    "day 2": {{
      "Plan": "Detailed paragraph for the second day, continuing logically from day 1.",
      "locations": [
        {{
          "name": "Location/Attraction name",
          "latitude": 0.0,
          "longitude": 0.0
        }}
      ]
    }},
    "...": {{
      "Plan": "Continue for all {total_days} days.",
      "locations": [
        {{
          "name": "Location/Attraction name",
          "latitude": 0.0,
          "longitude": 0.0
        }}
      ]
    }}
  }},
  "budget_breakdown": {{
    "accommodation": "Approximate cost in USD",
    "food": "Approximate cost in USD",
    "activities": "Approximate cost in USD",
    "transportation": "Approximate cost in USD",
    "miscellaneous": "Approximate cost in USD"
  }}
}}


### Additional Instructions for the Model:
1. Each day’s “Plan” must read like a realistic travel journal entry — informative, concise, and story-like.
2. Ensure every day connects logically (travel between cities, rest days if needed, etc.).
3. Include specific attractions, neighborhoods, or food experiences.
4. Stay consistent with traveler type: "{tour_requests[0].type}" and budget: "{tour_requests[0].budget}".
5. Avoid lists or subheadings — only use narrative paragraphs inside “Plan”.
6. Respond **only** with valid JSON, no explanations or markdown.
    """

    return prompt

@app.get("/")
async def root():
    return {"message": "Tour Planning API - Use POST /plan-tour to generate tour plans"}

@app.post("/plan-tour", response_model=TourPlanResponse)
async def plan_tour(tour_requests: List[TourRequest]):
    """
    Generate a tour plan based on multiple tour requests
    """
    if not tour_requests:
        raise HTTPException(status_code=400, detail="At least one tour request is required")
    
    # Validate OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    try:
        print(f"Processing tour request for {len(tour_requests)} destinations")
        tour_plan = generate_tour_plan(tour_requests)
        
        return TourPlanResponse(
            tour_plan=tour_plan,

        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in plan_tour: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Tour Planning API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2000)
