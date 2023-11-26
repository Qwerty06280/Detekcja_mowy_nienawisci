# import libraries
# pip install google-api-python-client
import os
import pandas as pd
from googleapiclient.discovery import build
from typing import List
from urllib.parse import urlparse, parse_qs

def get_id(url: str) -> str:
    """
    Extracts youtube video ID from a complete link
    :param url: link to youtube video
    :return: extracted video ID
    """
    u_pars = urlparse(url)
    quer_v = parse_qs(u_pars.query).get('v')
    if quer_v:
        return quer_v[0].strip()
    pth = u_pars.path.split('/')
    if pth:
        return pth[-1].strip()

def extract_comments(video_id: str, limit: int = 10) -> List:
    """
    Extracts comments from a video
    :param video_id: video_id_, value returned by get_id() function
    :param limit: limit the max number of comments
    :return: extracted comments -> List
    """
    # function that extracts comments from a video
    # empty list for storing replies
    replies = []
    all_replies = []
    # creating youtube resource object
    yt_api_key = os.environ.get('YT_API_KEY')
    yt_connection = build('youtube', 'v3', developerKey=yt_api_key)
    # for indexing purpose
    limit = limit -1
    if limit < 100:
        max_Results = limit
    else:
        max_Results = 100
    # retrieve youtube video results
    video_response = yt_connection.commentThreads().list(
        part='snippet,replies',
        videoId=video_id,
        maxResults=max_Results  # MAX 100
    ).execute()
    # iterate video response
    while video_response:
        # extracting required info
        # from each result object
        for item in video_response['items']:
            # Extracting comments
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            all_replies.append(comment)
            # counting number of reply of comment
            replycount = item['snippet']['totalReplyCount']
            # if reply is there
            if replycount > 0:
                # iterate through all reply
                for reply in item['replies']['comments']:
                    # Extract reply
                    reply = reply['snippet']['textDisplay']
                    # Store reply is list
                    replies.append(reply)
                    all_replies.append(reply)
            # empty reply list
            replies = []
        if len(all_replies) > limit:
            break
        elif limit - len(all_replies) < 100:
            max_Results2 = (limit - len(all_replies))
        else:
            max_Results2 = 100
        # Again repeat
        if 'nextPageToken' in video_response:
            video_response = yt_connection.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                maxResults=max_Results2,
                pageToken=video_response['nextPageToken']
            ).execute()
        else:
            break
    return all_replies

def yt_extract(url: str = 'https://www.youtube.com/watch?v=v7CQkivQNQI', limit: int = 10):
    """
    Takes url as an argument and extract various data from youtube video
    :param url: link to youtube video
    :return: (dataframe with extracted comments, title of the video)
    """
    youtube_link = url
    # extract video_id with function get_id()
    video_id = get_id(youtube_link)
    # Get API key and connect to Youtube API
    yt_api_key = os.environ.get('YT_API_KEY')
    yt_connection = build('youtube', 'v3', developerKey=yt_api_key)

    # Extract comments with function extract_comments()
    result = extract_comments(video_id, limit = limit)

    # get the title of the video
    response_title = yt_connection.videos().list(
        part='snippet',
        id=video_id
    ).execute()

    video_title = response_title['items'][0]['snippet']['title']
    df_comments = pd.DataFrame({"Comment": result})
    return df_comments, video_title

## test
# comments, title = yt_extract(limit = 4)
# print("Video title:\n{}\n\nNumber of comments that have been found: \n{}".format(title, len(comments)))
# print(comments.tail(10))

