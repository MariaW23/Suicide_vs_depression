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
            # add posts into the list
            output_list.extend(json["data"]["children"])
            after = json["data"]["after"]

        else:
            print(res.status_code)
            break

        # avoid getting rate limited
        time.sleep(5)
    
    print("<<<SCRAPING COMPLETED>>>")
    print(f"Number of posts downloaded: {len(output_list)}")

    # utilize set() and getting actual posts' names to get unique posts
    print("Number of unique posts: {}".format(len(set([p["data"]["name"] for p in output_list]))))

    return output_list

def create_unique_list(original_list: list) -> list:
    data_name_set = set()
    unique_list = []
    for i in range(len(original_list)):
        curr_data = original_list[i]["data"]
        if curr_data["name"] not in data_name_set:
            data_name_set.add(curr_data["name"])
            unique_list.append(curr_data)
    print(f"LIST NOW CONTAINS {len(unique_list)} UNIQUE SCRAPED POSTS")
    return unique_list


suicide_data = reddit_scrape(SUICIDE_URL, 80)
suicide_data_unique = create_unique_list(suicide_data)
suicide_watch = pd.DataFrame(suicide_data_unique)
# Add a column indicating the posts are from the SuicideWatch subreddit
suicide_watch["is_suicide"] = 1
suicide_watch.to_csv('suicide_watch.csv', index=False)


depression_data = reddit_scrape(DEPRESSION_URL, 80)
depression_data_unique = create_unique_list(depression_data)
depression = pd.DataFrame(depression_data_unique)
depression["is_suicide"] = 0
depression.to_csv('depression.csv', index=False)

# create combined CSV with selected columns
depression = pd.read_csv('depression.csv')
suicide_watch = pd.read_csv('suicide_watch.csv')
dep_columns = depression[["title", "selftext", "author", "num_comments", "is_suicide", "url"]]
sui_columns = suicide_watch[["title", "selftext", "author", "num_comments", "is_suicide", "url"]]
# use axis=0 so as concatination done via adding rows, 1 would mean having side by side data structure
# ignore index so that there's no duplicate indices
combined_data = pd.concat([dep_columns, sui_columns], axis=0, ignore_index=True)

# Fill in missing selftext values with "emptypost"
combined_data.fillna({"selftext": "emptypost"}, inplace=True)
combined_data.head()

# Display the number of missing values in each column.
combined_data.isnull().sum()

# saving combined CSV
combined_data.to_csv('suicide_vs_depression.csv', index = False)
