# All credit goes to: http://opensource.box.com/box-python-sdk/tutorials/intro.html
# Import two classes from the boxsdk module - Client and OAuth2
from boxsdk import Client, OAuth2
from constants import CONFIG_FILE, CS231N_PROJECT_FOLDER, MODEL_FOLDER, SYNC_DIRECTORIES
import os
import glob
import threading


def get_user(client):
    '''
    Prints out current user
    '''
    acct = client.user().get()
    print(acct.name)
    print(acct.login)
    print(acct.avatar_url)


def exists(root, subdir_name):
    '''
    Whether a subdirectory exists. If so, returns the folder object.
    
    Keyword arguments:
    root (Box folder) -- the root directory to search in.
    subdir_name (string) -- the name of the file/folder to find.
    
    Returns:
    exists (bool) -- whether the given name exists in root.
    subdir (Box folder) -- the actual subdirectory (or `None` if
        `exists` is False).
    '''
    subdirectories = root.get_items()
    # Iterates through stuff in root folder
    subdir = None
    for thing in subdirectories:
        if thing.name == subdir_name:
            return True, thing
    return False, None


def locate_folder(root, name):
    '''
    Wrapper for "locating" a folder. Creates one if doesn't exist.
    '''
    if not exists(root, name)[0]:
        create_subfolder = root.create_subfolder(name)

    found, folder = exists(root, name)
    assert(found)
    return folder


def sync(client):
    '''
    Synchronizes all folders from constants.SYNC_DIRECTORIES
    given a logged-in client.
    '''
    print('Begin syncing items to Box...')
    # Root Box directory
    root_folder = client.folder('0')
    
    # Grabs the online cs231n folder
    cs231n_folder = locate_folder(root_folder, CS231N_PROJECT_FOLDER)
    
    # Loops through local directories needing to be synced up
    for local_dir_name in SYNC_DIRECTORIES:
        sync_helper(cs231n_folder, local_dir_name)
    print('Finished syncing items to Box!')

        
def sync_helper(root, full_local_dir_name):
    '''
    Recursively copies over all things in local_dir_name to root.
    Precondition: Substring of full_local_dir_name after the final
    forward slash is the subfolder we want to make in root.
    '''
    def contains(root_items, name):
        '''
        Speeds up sync_helper by sending fewer requests.
        '''
        for thing in root_items:
            if thing.name == name:
                return True
        return False
    
    # Grabs subfolder + contents
    subfolder_name = get_item_name(full_local_dir_name)
    subfolder = locate_folder(root, subfolder_name)
    current_subfolder_items = list(subfolder.get_items()._items_generator()) # Hacky
    
    # Spawn one worker for each subdirectory
    all_workers = []
    for subsubpath in glob.glob(full_local_dir_name + '/*'):
        if os.path.isdir(subsubpath):
            worker = threading.Thread(target=sync_helper, args=(subfolder, subsubpath))
            all_workers.append(worker)
            worker.start()
        elif not contains(current_subfolder_items, get_item_name(subsubpath)):
            print('Uploading', subsubpath)
            subfolder.upload(subsubpath)

    # Wait for all child threads to finish
    for worker in all_workers:
        worker.join()

def get_item_name(full_local_path):
    '''
    Grabs just the folder/filename from the full path.
    '''
    return full_local_path[full_local_path.rfind('/') + 1:]

        
def main():
    # Read app info from text file
    with open(CONFIG_FILE, 'r') as app_cfg:
        CLIENT_ID = app_cfg.readline().strip()
        CLIENT_SECRET = app_cfg.readline().strip()
        ACCESS_TOKEN = app_cfg.readline().strip()

    # Create OAuth2 object. It's already authenticated, thanks to the developer token.
    oauth2 = OAuth2(CLIENT_ID, CLIENT_SECRET, access_token=ACCESS_TOKEN)

    # Create the authenticated client
    client = Client(oauth2)
    
    # Begin syncing everything to Box
    sync(client)
    
    
if __name__ == '__main__':
    main()