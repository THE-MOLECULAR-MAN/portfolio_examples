#!/bin/bash
# Tim H 2022
# real, current documentation page: https://www.reddit.com/dev/api
# working script to dump all comments and posts by a single user
# example usage: ./dump-user-reddit-api.sh "user123"
# relies on external file named "reddit-api-creds" for API creds. Takes this format:
#   export APP_NAME="whatever it is named in Reddit"
#   export CLIENT_ID="REDACTED"
#   export CLIENT_SECRET="REDACTED"
#   export DEVELOPER_REDDIT_USERNAME="REDACTED"

# References:
#   https://github.com/reddit-archive/reddit/wiki/OAuth2#application-only-oauth

# TODO:
#    follow certain domain links and download those files for submissions or files included in comments?
set -e

DUMP_REDDIT_USERNAME="$1"
CURL_CONFIG_FILE="curl_config.txt"

# load variables from another file
# shellcheck disable=SC1091
source "reddit-api-creds"

# you should not need to change these:
MAX_RESULTS_LIMIT="100"
SLEEP_TIME_SEC="2"
APP_VERSION="0.2.1"

# must be last:
USER_AGENT_HEADER="User-Agent: OSX:$APP_NAME:v$APP_VERSION (by /u/$DEVELOPER_REDDIT_USERNAME)"

################################################################################
#		FUNCTION DEFINITIONS
################################################################################
reddit_api_authenticate_Application_Only_OAuth_webapps_and_scripts () {
    #Application Only OAuth
    # App-only OAuth token requests never receive a refresh_token
    # attempt to authenticate and get a response that includes the access token
    echo "Authenticating to Reddit API to get new access token..."
    RESP=$(curl --silent -H "$USER_AGENT_HEADER" \
        -d "grant_type=client_credentials" \
        --user "$CLIENT_ID:$CLIENT_SECRET" \
        https://www.reddit.com/api/v1/access_token)
    sleep "$SLEEP_TIME_SEC"
    # extract the access token
    ACCESS_TOKEN=$(extract_access_token "$RESP")
}

extract_access_token() {
    # converts a JSON text blob into just the access token value
    JSON_RESPONSE="$1"
    echo "$JSON_RESPONSE" | jq -r .access_token
}

create_curl_config_file () {
    # create a config file to simplify curl future calls
    # tested and verified that passing the User-Agent header manually instead 
    # of using curl's built in flag is fine
    # do not adjust indentation here
    cat << EOF > "$CURL_CONFIG_FILE"
-H "$USER_AGENT_HEADER"
-H "Authorization: bearer $ACCESS_TOKEN"
-H "Accept: application/json"

EOF
}

dump_redditor () {
    # the real workhorse, what goes through and dumps a user's account
    # loops through until there are no more parts
    for TYPE_ITER in submitted comments
    do
        PART_COUNTER=1
        NEXT_AFTER=""

        until [[ "$NEXT_AFTER" == "null" ]]
        do
            OUTPUT_FILENAME="Reddit-user-dump-$DUMP_REDDIT_USERNAME-$TYPE_ITER-pt$PART_COUNTER.json"

            echo "Dump user $DUMP_REDDIT_USERNAME $TYPE_ITER part $PART_COUNTER..."
            curl --silent -o "$OUTPUT_FILENAME" -K "$CURL_CONFIG_FILE" "https://oauth.reddit.com/user/$DUMP_REDDIT_USERNAME/$TYPE_ITER?limit=$MAX_RESULTS_LIMIT&after=$NEXT_AFTER"
            NEXT_AFTER=$(jq --raw-output '.data.after' "$OUTPUT_FILENAME" )
            PART_COUNTER=$((PART_COUNTER+1))
            sleep "$SLEEP_TIME_SEC"
        done
    done
    
    # convert json to csv, only relevant columns
    # https://richrose.dev/posts/linux/jq/jq-json2csv/
    # convert all (multiple) comments JSON files into a single CSV
    jq -r '.data.children[].data | [.created_utc,.subreddit_name_prefixed,.link_title,.link_url,.link_author,.author,.body,.edited] | @csv' Reddit-user-dump-"$DUMP_REDDIT_USERNAME"-comments-pt*.json > Reddit-user-dump-"$DUMP_REDDIT_USERNAME"-comments-combined.csv

    # convert all (multiple) submissions (posts) JSON files into a single CSV
    jq -r '.data.children[].data | [.created_utc,.subreddit_name_prefixed,.title,.selftext,.permalink,.domain,.url] | @csv' Reddit-user-dump-"$DUMP_REDDIT_USERNAME"-submitted-pt*.json > Reddit-user-dump-"$DUMP_REDDIT_USERNAME"-submitted-combined.csv
    
    # intentionally do not put double quotes on next line
    sha256sum Reddit-user-dump-"$DUMP_REDDIT_USERNAME"*.json > "Reddit-user-dump-$DUMP_REDDIT_USERNAME-hashes.sha256"
    NOW=$(date +"%Y_%m_%d_%I_%M_%p_%z")
    ZIP_FILENAME="Reddit-user-dump-$DUMP_REDDIT_USERNAME-$NOW.zip"
    zip -q -m "$ZIP_FILENAME" Reddit-user-dump-"$DUMP_REDDIT_USERNAME"*.json Reddit-user-dump-"$DUMP_REDDIT_USERNAME"*.sha256 Reddit-user-dump-"$DUMP_REDDIT_USERNAME"*.csv
}

################################################################################
#		MAIN PROGRAM
################################################################################
reddit_api_authenticate_Application_Only_OAuth_webapps_and_scripts
create_curl_config_file
dump_redditor

rm -f "$CURL_CONFIG_FILE"
echo "finished dump-user-reddit-api.sh successfully"
