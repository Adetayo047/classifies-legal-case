from api import app  # Import your Flask app

with app.test_client() as client:
    response = client.post(
        '/pred_label',
        json={"full_report": "In absence of any specific allegation against the investigating agency acting illegally and contrary to the detriment of the defence and the State, the Sessions Judge has no authority or reason to have directed the Superintendent of Police to transfer the investigation of the case from one Investigating Officer to the other"}
    )
    print("Status code:", response.status_code)
    print("Response JSON:", response.get_json())
