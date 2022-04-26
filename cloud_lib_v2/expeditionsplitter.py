from cloud_lib_v2.expedition import Expedition

class ExpeditionSplitter():
    def __init__(self, expedition: Expedition, core_number: int):

        expedition.df_events = expedition.df_events()
        self.batches = []
