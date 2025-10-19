import dagster as dg
import dagster_slack

slack_provider = dagster_slack.SlackResource(token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN"))
