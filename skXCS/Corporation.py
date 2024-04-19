class Corporation:
    def __init__(self):
        self.corp={}
    
    def collate(self,keyId,itemId):
        self.corp.setdefault(keyId,[]).append(itemId)