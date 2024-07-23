from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
import webbrowser
from wsgiref.simple_server import make_server
import urllib.parse

# Update scope to allow managing all Drive files
SCOPES = ['https://www.googleapis.com/auth/drive']
CLIENT_SECRETS_FILE = 'secrets/client_secrets.json'


def authorize():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri='http://localhost:5001'
    )

    auth_url, _ = flow.authorization_url(prompt='consent')

    print(f"Please visit this URL to authorize the application: {auth_url}")
    webbrowser.open(auth_url)

    def callback(environ, start_response):
        params = urllib.parse.parse_qs(environ['QUERY_STRING'])
        if 'code' in params:
            flow.fetch_token(code=params['code'][0])
            start_response('200 OK', [('Content-Type', 'text/plain')])
            return [b'Authentication successful! You can close this window.']
        start_response('400 Bad Request', [('Content-Type', 'text/plain')])
        return [b'Authentication failed.']

    with make_server('', 5001, callback) as httpd:
        print("Waiting for callback on port 5001...")
        httpd.handle_request()

    credentials = flow.credentials

    with open('token.json', 'w') as token_file:
        token_file.write(credentials.to_json())

    print("Authorization successful! Token saved to token.json")


if __name__ == '__main__':
    authorize()
