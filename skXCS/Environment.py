
from skXCS.DataManagement import DataManagement
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
import numpy as np
class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None

    def insert_at_end(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
            new_node.prev = current

    def display(self):
        current = self.head
        while current:
            print(current.data, end=" <-> ")
            current = current.next
        print("None")


class Environment:
    def build_linked_list_from_list(self,lst):
        dll = DoublyLinkedList()
        for item in lst:
            dll.insert_at_end(item)
        return dll
    def __init__(self,X,y,xcs):
        self.dataRef = 0
        self.formatData = DataManagement(X,y,xcs)
        self.max_payoff = xcs.max_payoff

        self.currentTrainState = self.formatData.trainFormatted[0][self.dataRef]
        self.currentTrainPhenotype = self.formatData.trainFormatted[1][self.dataRef]    
        self.trainState = self.build_linked_list_from_list(self.formatData.trainFormatted[0])
        self.trainPhenotype = self.build_linked_list_from_list(self.formatData.trainFormatted[1])

        self.currentTrainState = self.trainState.head
        self.currentTrainPhenotype = self.trainPhenotype.head

        self.balancedXIndex=np.arange(len(self.formatData.trainFormatted[0]))
        self.balancedXIndex,self.balacedLbl=rus.fit_resample(self.balancedXIndex.reshape(-1, 1),self.formatData.trainFormatted[1])

        self.balancedXIndex=np.sort(self.balancedXIndex.reshape(   1,-1)[0])
        self.indexRef=0

        while self.dataRef != self.balancedXIndex[self.indexRef]:
            self.dataRef=self.dataRef+1
            self.currentTrainState=self.currentTrainState.next
            self.currentTrainPhenotype=self.currentTrainPhenotype.next
    def getTrainState(self):
        return self.currentTrainState

    def newInstance(self):
        if self.dataRef < self.formatData.numTrainInstances-1 and self.indexRef < self.balancedXIndex.shape[0]-1:
            self.indexRef+=1

            while self.dataRef != self.balancedXIndex[self.indexRef]:
                self.dataRef=self.dataRef+1
                self.currentTrainState=self.currentTrainState.next
                self.currentTrainPhenotype=self.currentTrainPhenotype.next
            #self.currentTrainState = self.formatData.trainFormatted[0][self.dataRef]
            #self.currentTrainPhenotype = self.formatData.trainFormatted[1][self.dataRef]

            ########
            #self.currentTrainState = self.currentTrainState.next
            #self.currentTrainPhenotype = self.currentTrainPhenotype.next
        else:
            self.resetDataRef()
            
    def nextInstance(self):
        if self.dataRef < self.formatData.numTrainInstances-1:
            return self.formatData.trainFormatted[0][self.dataRef+1],self.formatData.trainFormatted[1][self.dataRef+1]
        return None,None
    def prevInstance(self):
        if self.dataRef>0:
            return self.formatData.trainFormatted[0][self.dataRef-1],self.formatData.trainFormatted[1][self.dataRef-1]
        return None,None

    def resetDataRef(self):
        self.dataRef = 0
        self.currentTrainState = self.formatData.trainFormatted[0][self.dataRef]
        self.currentTrainPhenotype = self.formatData.trainFormatted[1][self.dataRef]

        self.indexRef=0
        self.currentTrainState = self.trainState.head
        self.currentTrainPhenotype = self.trainPhenotype.head

    def executeAction(self,action):
        if action == self.currentTrainPhenotype.data:
            return self.max_payoff
        return 0

