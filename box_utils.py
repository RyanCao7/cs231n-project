# All credit goes to: http://opensource.box.com/box-python-sdk/tutorials/intro.html
# Import two classes from the boxsdk module - Client and OAuth2
import boxsdk
import os
import subprocess
import glob
import threading
from boxsdk import Client, OAuth2
from constants import CONFIG_FILE, CS231N_PROJECT_FOLDER, MODEL_FOLDER, SYNC_DIRECTORIES
from datetime import datetime


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


def sync():
    '''
    Synchronizes all folders from constants.SYNC_DIRECTORIES
    given a logged-in client.
    '''
    client = get_client()
    print('Begin syncing items to Box...')
    # Root Box directory
    root_folder = client.folder('0')
    
    # Grabs the online cs231n folder
    cs231n_folder = locate_folder(root_folder, CS231N_PROJECT_FOLDER)
    
    # Loops through local directories needing to be synced up
    for local_dir_name in SYNC_DIRECTORIES:
        sync_helper(cs231n_folder, local_dir_name)
    print('Finished syncing items to Box!')


def contains(root_items, name):
    '''
    Speeds up sync_helper by sending fewer requests.
    '''
    for thing in root_items:
        if get_filename(thing.name) == get_filename(name):
            return True, thing.name
    return False, None
    
    
def sync_helper(root, full_local_dir_name):
    '''
    Recursively copies over all things in local_dir_name to root.
    Precondition: Substring of full_local_dir_name after the final
    forward slash is the subfolder we want to make in root.
    '''
    
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
        else:
            exists, old_filename = contains(current_subfolder_items, get_item_name(subsubpath))
            if not exists:
                print('Uploading', subsubpath)
                subfolder.upload(subsubpath)
                continue
            if should_update(old_filename, get_item_name(subsubpath)):
                file = None
                for item in current_subfolder_items:
                    if get_filename(item.name) == get_filename(get_item_name(subsubpath)):
                        file = item
                print('Updating', subsubpath)
                file.update_contents(subsubpath)

    # Wait for all child threads to finish
    for worker in all_workers:
        worker.join()

        
def convert():
    '''
    Converts all old filenames into timestamped ones (locally).
    SHOULD ONLY BE USED ONCE!!!
    '''
    for folder_name in SYNC_DIRECTORIES:
        convert_helper(folder_name, constants.get_cur_time())
        

def convert_helper(folder_name, cur_time, convert_back=False):
    '''
    Recursively converts all local files (not dirs) to new
    timestamped format.
    '''
    for item in glob.glob(folder_name + '/*'):
        if os.path.isdir(item):
            convert_helper(item, cur_time)
        else:
            filename = get_item_name(item).split('.')[0]
            extension = '.'.join(get_item_name(item).split('.')[1:])
            if convert_back:
                timestampped_name = filename[:filename.rfind('~')]
                timestampped_name = timestampped_name + '.' + extension
            else:
                timestampped_name = filename + '~' + cur_time + '.' + extension
            _ = subprocess.check_output(['mv', item, folder_name + '/' + timestampped_name])
            print('converted', filename, 'to', timestampped_name)
    
        
def get_filename(full_filename):
    '''
    Removes the trailing timestamp.
    full_filename format: [filename]~MM-DD|HH:MM:SS.[extension]
    '''
    return full_filename[:full_filename.find('~')] + full_filename[full_filename.find('.'):]
        
        
def get_item_name(full_local_path):
    '''
    Grabs just the folder/filename from the full path.
    '''
    return full_local_path[full_local_path.rfind('/') + 1:]


def should_update(old_filename, new_filename):
    '''
    Checks the timestamps on old_filename and new_filename
    to see whether we should update the file.
    '''
    old_timestamp = old_filename.split('~')[1].split('.')[0]
    new_timestamp = new_filename.split('~')[1].split('.')[0]
    return value_secs(old_timestamp) < value_secs(new_timestamp)


def value_secs(timestamp):
    '''
    Grabs total number of seconds from timestamp string
    in the form MM-DD|HH:MM:SS.
    
    Note: We assume 30 days per month.
    '''
    total_secs = 0
    month_day, time = timestamp.split('|')
    month_day = month_day.split('-')
    time = time.split(':')
    total_secs += int(month_day[0]) * 2592000 + int(month_day[1] * 86400)
    return total_secs + 3600 * int(time[0]) + 60 * int(time[1]) + int(time[2])
    

def sync_download():
    '''
    Syncs the folders in cs231n_folder to local filesystem.
    '''
    client = get_client()
    print('Begin syncing items from Box...')
    # Root Box directory
    root_folder = client.folder('0')
    
    # Grabs the online cs231n folder
    cs231n_folder = locate_folder(root_folder, CS231N_PROJECT_FOLDER)
    sync_download_helper('', cs231n_folder)
    
    
def should_local_update(local_folder, box_file_name):
    '''
    Given a local folder and box file name, checks to see whether
    we should overwrite the local state copy.
    '''
    for full_path in glob.glob(local_folder + '*'):
        if get_filename(box_file_name) == get_filename(get_item_name(full_path)):
            return should_update(get_item_name(full_path), box_file_name)
    return True
    
    
def sync_download_helper(local_path, box_folder):
    '''
    Recursive helper function to download Box folder tree.
    '''
    # Multithreaded downloading - one thread per directory.
    all_workers = []
    for item in box_folder.get_items():
        if type(item) == boxsdk.object.folder.Folder:
            new_path = local_path + item.name + '/'
            if not os.path.isdir(new_path):
                os.makedirs(new_path)
            worker = threading.Thread(target=sync_download_helper, args=(new_path, item))
            all_workers.append(worker)
            worker.start()
        else:
            if should_local_update(local_path, item.name):
                print('Downloading', item.name, 'from Box...')
                local_file = open(local_path + item.name, 'wb')
                item.download_to(local_file)
                local_file.close()


    # Wait for all download threads to finish
    for worker in all_workers:
        worker.join()
    
    
def get_client():
    '''
    Grabs client object from hidden config file
    (see constants.py for CONFIG_FILE location).
    '''
    
    # Read app info from text file
    with open(CONFIG_FILE, 'r') as app_cfg:
        CLIENT_ID = app_cfg.readline().strip()
        CLIENT_SECRET = app_cfg.readline().strip()
        ACCESS_TOKEN = app_cfg.readline().strip()

    # Create OAuth2 object. It's already authenticated, thanks to the developer token.
    # TODO: Make this not depend on the developer token?!!
    oauth2 = OAuth2(CLIENT_ID, CLIENT_SECRET, access_token=ACCESS_TOKEN)

    # Create the authenticated client
    client = Client(oauth2)
    
    return client


def upload_single(local_file_path):
    '''
    Takes a local file and sticks it in the correct corresponding
    Box folder.
    '''
    client = get_client()
    # Root Box directory
    folder = client.folder('0')
    
    # Grabs the online cs231n folder
    folder = locate_folder(folder, CS231N_PROJECT_FOLDER)
    
    # Keeps a copy to reference later for uploading/updating.
    full_path = str(local_file_path)
    
    # Drill down to the correct folder
    while '/' in local_file_path:
        next_root = local_file_path[:local_file_path.find('/')]
        folder = locate_folder(folder, next_root)
        local_file_path = local_file_path[local_file_path.find('/') + 1:]
        
    current_folder_items = list(folder.get_items()._items_generator()) # Hacky
    exists, old_filename = contains(current_folder_items, local_file_path)
    if not exists:
        print('Uploading', full_path)
        folder.upload(full_path)
    else:
        file = None
        for item in current_folder_items:
            if get_filename(item.name) == get_filename(local_file_path):
                file = item
        print('Updating', full_path)
        file.update_contents(full_path)


def main():
    pass
    # Grabs client to login from
#     client = get_client()
    
    # Converts local files to timestampped format
#     convert()
    
    # Begin syncing everything to Box
#     sync_download()
#     sync()
    
    
if __name__ == '__main__':
    main()