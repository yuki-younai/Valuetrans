
import random
import json

with open('src/person.json', 'r') as file:
    person_info = json.load(file)

def generate_persona_occupation_description(personas_per_question):
    
    personas = []
    for _ in range(personas_per_question):
        #character
        age=random.randint(person_info['character']["age"][0],person_info['character']["age"][1])
        gender=random.choice(person_info['character']['gender'])
        occupation=random.choice(person_info['character']['occupation'])
        country=random.choice(person_info['character']['country']),
        political_orientation=random.choice(person_info['character']['political_orientation'])
        marital_status=random.choice(person_info['character']['marital_status'])
        children_status=random.choice(person_info['character']['has_children'])
        ethnicity=random.choice(person_info['character']['ethnicity'])
        religion=random.choice(person_info['character']['religion'])
        personality_trait=random.choice(person_info['character']['personality_traits'])
        pet_ownership=random.choice(person_info['character']['pet_ownership'])
        active_level=random.choice(person_info['character']["social_media_usage"]["active_level"])
        usage_style=random.choice(person_info['character']["social_media_usage"]["usage_style"])
        social_media_preference=random.choice(person_info['character']["social_media_usage"]["social_media_preference"])
        health_status=random.choice(person_info['character']["health"]["health_status"])
        mental_health_status=random.choice(person_info['character']["health"]["mental_health_status"])
        #background_story
        family_education=random.choice(person_info["background_story"]["family_education"])
        life_turning_points=random.choice(person_info["background_story"]["life_turning_points"])
        conflicts_or_challenges=random.choice(person_info["background_story"]["conflicts_or_challenges"])
        #social_cultural_environment
        geographical_location=random.choice(person_info["social_cultural_environment"]["geographical_location"])
        cultural_features=random.choice(person_info["social_cultural_environment"]["cultural_features"])

        for key, value in occupation.items():
            job = key
            income_range = value
        income=random.randint(income_range[0], income_range[1])
        
        description = f"""
        You are a {age} years old, and your gender is {gender}. you currently work as {job} in {country}, with an annual income of approximately {income}.Your political inclination is {political_orientation}.Your marital status is {marital_status}, and you have {children_status}.You belong to {ethnicity} and practice {religion}. As a {personality_trait} person, You have my own views and attitudes in life.Your pet ownership situation is {pet_ownership}.Regarding social media, You {active_level} {usage_style} prefer to {social_media_preference}.About your health, your current physical health status is {health_status}, and your mental state is {mental_health_status}.Your family education background has a profound influence on you. Your parents {family_education}. There have been several key turning points in your life, the most significant being {life_turning_points}. Of course, life inevitably presents conflicts and challenges. For example, you felt confused and helpless because of {conflicts_or_challenges}.You currently live in {geographical_location}.Here, the local cultural characteristics are {cultural_features}.
        """
        description = description.strip() 
        personas.append({
            'description': description,
            'sex': gender,
            'age': age,
            'coutry_of_origin':country,
            'country_of_residence': political_orientation,
            'income_level': income,
            'marrital_status': marital_status,
            'have_children': children_status
        })

    return personas
