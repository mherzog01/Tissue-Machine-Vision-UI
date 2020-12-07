# https://stackoverflow.com/questions/57772453/login-on-colab-with-gcloud-without-service-account
import httplib2
import json

from google.colab import auth
from oauth2client import GOOGLE_REVOKE_URI, GOOGLE_TOKEN_URI, client
from oauth2client.client import GoogleCredentials
#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive

class ColabAuthGCP():
  def __init__(self, run_auth=True):
    self.auth_key = {
      "client_id": "32555940559.apps.googleusercontent.com",
      "client_secret": "ZmssLNjJy2998hD4CTg2ejr2",
      "refresh_token": "1//06pZumWkLoVZECgYIARAAGAYSNgF-L9IryqSVFw3_nPE3JeJ-XDgN9O1IMcF2i9kbZI5IsJV6MisaUr_XLckHEgiDc8fkBZUZbQ"
    }
    if run_auth:
      self.auth_user()

  def auth_user(self):
    # TODO Move print statements to logger
    print('Authentication begins')
    credentials = client.OAuth2Credentials(
        access_token=None,
        client_id=self.auth_key['client_id'],
        client_secret=self.auth_key['client_secret'],
        refresh_token=self.auth_key['refresh_token'],
        token_expiry=None,
        token_uri=GOOGLE_TOKEN_URI,
        user_agent=None,
        revoke_uri=GOOGLE_REVOKE_URI)

    credentials.refresh(httplib2.Http())
    credentials.authorize(httplib2.Http())
    cred = json.loads(credentials.to_json())
    cred['type'] = 'authorized_user'

    with open('adc.json', 'w') as outfile:
      json.dump(cred, outfile)

    auth.authenticate_user()
    #gauth = GoogleAuth()
    #gauth.credentials = credentials
    #drive = GoogleDrive(gauth)
    print('Authentication complete')

