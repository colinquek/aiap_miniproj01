# aiap_miniproj01
| Attribute | Description |
|---|---|
| ID | Unique ID for each news article |
| URL | URL of the news article |
| timedelta | Number of days between the article's publication and the dataset acquisition |
| weekday | The day of the week when the article was published |
| n_tokens_title | Number of words in the title |
| n_tokens_content | Number of words in the content |
| n_unique_tokens | Rate of unique words in the content |
| n_non_stop_words | Rate of non-stop words in the content |
| n_non_stop_unique_tokens | Rate of unique non-stop words in the content |
| num_hrefs | Number of links |
| num_self_hrefs | Number of links to other articles published by the news website |
| num_imgs | Number of images |
| num_videos | Number of videos |
| n_comments | Number of comments by readers |
| average_token_length | Average length of the words in the content |
| data_channel | The data channel of the article |
| self_reference_min_shares | Minimum shares of referenced articles in the news website |
| self_reference_max_shares | Maximum shares of referenced articles in the news website |
| self_reference_avg_shares | Average shares of referenced articles in the news website |
| num_keywords | Number of keywords in the metadata |
| kw_min_min | Minimum shares any article obtained with the worst performing keyword in the article |
| kw_max_min | Maximum shares any article obtained with the worst performing keyword in the article |
| kw_avg_min | Average shares any article obtained with the worst performing keyword in the article |
| kw_min_max | Minimum shares any article obtained with the best performing keyword in the article |
| kw_max_max | Maximum shares any article obtained with the best performing keyword in the article |
| kw_avg_max | Average shares any article obtained with the best performing keyword in the article |
| kw_min_avg | Minimum shares any article obtained with the median performing keyword in the article |
| kw_max_avg | Maximum shares any article obtained with the median performing keyword in the article |
| kw_avg_avg | Average shares any article obtained with the median performing keyword in the article |
| shares | Number of times the article was shared (Target) |

# EDA
1. num_videos, 