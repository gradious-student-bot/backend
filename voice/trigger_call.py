import asyncio
import os
from dotenv import load_dotenv
from livekit import api
from config import LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
 
load_dotenv()

async def make_call():
   
    lk_api = api.LiveKitAPI(
        url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET
    )
    
    try:
        print("  Sending SIP Invite to Exotel...")
         
        request = api.CreateSIPParticipantRequest(
            sip_trunk_id="ST_iPHZ9DXZktip",  
            sip_call_to="08498992688",   
            room_name="my_test_room",
            participant_identity="student_phone"
        )
         
        await lk_api.sip.create_sip_participant(request)
        print(" Invite sent! The customer phone should ring in a few seconds.")
    
    except Exception as e:
        print(f" Failed to trigger call: {e}")
    finally:
        await lk_api.aclose()

if __name__ == "__main__":
    asyncio.run(make_call())




#airtable code [not tested]
# import os
# import asyncio
# from dotenv import load_dotenv
# from pyairtable import Api
# from livekit import api

# load_dotenv()

# async def trigger_calls():

#     print("Connecting to Airtable...")
#     airtable = Api(os.environ["AIRTABLE_API_KEY"])
#     table = airtable.table(os.environ["AIRTABLE_BASE_ID"], os.environ["AIRTABLE_TABLE_NAME"])

#     records = table.all()
#     print(f"Found {len(records)} total records in Airtable.")

#     lkapi = api.LiveKitAPI()
#     sip_client = api.SipClient(lkapi)
#     sip_trunk_id = os.environ["SIP_TRUNK_ID"]

#     for record in records:
#         fields = record.get('fields', {})
#         student_number = fields.get('student_number')
#         record_id = record.get('id')

#         if not student_number:
#             continue

#         clean_num = str(student_number).replace('+91', '').replace('-', '').replace(' ', '')
        
#         if not clean_num.startswith('0'):
#             clean_num = '0' + clean_num

#         room_name = f"call_{record_id}"

#         print(f"\n[DIALING] Triggering Exotel call to {clean_num} in room: {room_name}")

#         try:
          
#             await sip_client.create_sip_participant(
#                 sip_trunk_id=sip_trunk_id,
#                 sip_call_to=clean_num,
#                 room_name=room_name,
#                 participant_identity="student_phone"
#             )
#             print(" Call dispatched successfully to the telecom network!")
            
#             # TODO: Remove this 'break' once you want it to call the whole list at once
#             break 
            
#         except Exception as e:
#             print(f" Failed to trigger call: {e}")

#     # Close the API connection cleanly
#     await lkapi.aclose()

# if __name__ == "__main__":
#     asyncio.run(trigger_calls())