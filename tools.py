TOOLS = {
    "get_user_info": {
        "user_id": "string"
    },
    "send_email": {
        "recipient_email": "string",
        "subject": "string",
        "body": "string"
    },
    "generate_report": {
        "data": "list",
        "report_type": "string"
    },
    "fetch_weather": {
        "location": "string"
    },
    "search_web": {
        "query": "string",
        "num_results": "integer"
    },
    "translate_text": {
        "text": "string",
        "target_language": "string"
    },
    "analyze_sentiment": {
        "text": "string"
    },
    "summarize_article": {
        "article_text": "string",
        "summary_length": "integer"
    },
    "convert_currency": {
        "amount": "float",
        "from_currency": "string",
        "to_currency": "string"
    },
    "schedule_meeting": {
        "participants": "list",
        "meeting_time": "string",
        "agenda": "string"
    },
    "check_grammar": {
        "text": "string"
    },
    "create_container": {
        "container_name": "string",
        "image": "string",
        "ports": "list"
    },
}
