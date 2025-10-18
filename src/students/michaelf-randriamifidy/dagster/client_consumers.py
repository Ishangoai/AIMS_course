import dagster as dg
import dagster_slack

# class SlackResourceProvider:
#     _slack_resource = dagster_slack.SlackResource(
#         token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN")
#     )

#     @staticmethod
#     def get_resource():
#         return SlackResourceProvider._slack_resource

# slack_provider = SlackResourceProvider.get_resource()

slack_provider = dagster_slack.SlackResource(token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN"))