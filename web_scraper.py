import pandas as pd
import time
import requests

DEPRESSION_URL = "https://www.reddit.com/r/depression.json"
SUICIDE_URL = "https://www.reddit.com/r/SuicideWatch.json"

# create user-agent with random name to avoid 429 error (rate limit)
headers = {"User-agent" : "randomuser"}

def reddit_scrape(url, number_of_scrapes) -> list:
    after = None
    output_list = []
    seen_ids = set()
    total_downloaded = 0

    print(f"SCRAPING {url}\n--------------------------------------------------")
    print("<<<SCRAPING COMMENCED>>>")
    print(f"Downloading Batch {1} of {number_of_scrapes}...")
    # reddit default 25 posts/scrape, so scrape mulitple times
    for i in range(1, number_of_scrapes+1):
        # log batch number in console
        if i % 5 ==0:
            print(f"Downloading Batch {i} of {number_of_scrapes}...")
        
        # first scrape doesn't need to track last post's name
        params = {"after": after} if after else {}
        
        res = requests.get(url, headers=headers, params=params)
        
        if res.status_code == 200:
            json = res.json()
            total_downloaded += len(json["data"]["children"])
            print(f"Total number of posts downloaded: {total_downloaded}")

            batch_unique = 0
            for post in json["data"]["children"]:
                post_id = post["data"]["name"]
                if post_id not in seen_ids:
                    batch_unique += 1
                    seen_ids.add(post_id)
                    output_list.append(post["data"])
            print(f"Number of unique posts: {batch_unique}")
            after = json["data"]["after"]

            if not after:
                break
        else:
            print(f"Error {res.status_code}")
            break

        # avoid getting rate limited
        time.sleep(10)
    
    print("<<<SCRAPING COMPLETED>>>")
    print(f"Number of posts downloaded: {total_downloaded}")
    print(f"Number of unique posts: {len(seen_ids)}")
    return output_list

# Get suicide posts
# number_of_scrapes = 100
# suicide_data = reddit_scrape(SUICIDE_URL, number_of_scrapes)
# suicide_data = pd.DataFrame(suicide_data)
# # Add a column indicating the posts are from the SuicideWatch subreddit
# suicide_data["is_suicide"] = 1

# # raw data
# suicide_data.to_csv('data/suicide_raw.csv', index=False)

# # data with columns needed
# suicide_data = suicide_data[["selftext", "is_suicide"]]
# suicide_data.to_csv('data/suicide.csv', index=False)

# # cleaning data to avoid multi-lined posts
# suicide_data = pd.read_csv('data/suicide.csv', delimiter=',', quoting=1, encoding='utf-8')
# suicide_data['selftext'] = suicide_data['selftext'].str.replace('\n', '')
# suicide_data.to_csv('data/suicide_cleaned.csv', index=False)

# Get depression posts
# number_of_scrapes = 100
# depression_data = reddit_scrape(DEPRESSION_URL, number_of_scrapes)
# depression_data = pd.DataFrame(depression_data)
# # Add a column indicating the posts are from the SuicideWatch subreddit
# depression_data["is_suicide"] = 0

# # raw data
# depression_data.to_csv('data/depression_raw.csv', index=False)

# # data with columns needed
# depression_data = depression_data[["selftext", "is_suicide"]]
# depression_data.to_csv('data/depression.csv', index=False)

# cleaning data to avoid multi-lined posts
depression_data = pd.read_csv('data/depression.csv', delimiter=',', quoting=1, encoding='utf-8')
depression_data['selftext'] = depression_data['selftext'].str.replace('\n', '')
depression_data.to_csv('data/depression_cleaned.csv', index=False)


# # create combined CSV with selected columns

# depression = pd.read_csv('data/depression.csv')
# suicide_watch = pd.read_csv('data/suicide_watch.csv')
# dep_columns = depression[["selftext", "is_suicide"]]
# sui_columns = suicide_watch[["selftext", "is_suicide"]]
# # use axis=0 so as concatination done via adding rows, 1 would mean having side by side data structure
# # ignore index so that there's no duplicate indices
# combined_data = pd.concat([dep_columns, sui_columns], axis=0, ignore_index=True)

# # Fill in missing selftext values with "emptypost"
# combined_data.fillna({"selftext": "emptypost"}, inplace=True)
# combined_data.head()

# # Display the number of missing values in each column.
# combined_data.isnull().sum()

# # saving combined CSV
# combined_data.to_csv('data/data.csv', index = False)
