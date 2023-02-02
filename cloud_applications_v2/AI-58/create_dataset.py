import os

from cloud_lib_v2.expedition import Expedition

ai_58 = Expedition()
ai_58.init_using_json("AI-58-config.json")
ai_58.init_events()
ai_58.init_radiation()
ai_58.init_observation()

ai_58.merge_radiation_to_events()
ai_58.merge_observation_to_events()
ai_58.delete_outside_datetime("2021-08-11 10:00:00", "2021-08-11 09:59:00")
ai_58.compute_statistic_features()
ai_58.compute_correlation_to_previous()

print()
