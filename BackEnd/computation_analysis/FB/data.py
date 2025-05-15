class Data:
    def __init__(self):
        self.__apiKey="AIzaSyD7RKenqCBEHjvYNOGBWkZ1aJpulCtzPwU"
        self.__authDomain="genescope-c9328.firebaseapp.com"
        self.__projectId="genescope-c9328"
        self.__storageBucket="genescope-c9328.firebasestorage.app"
        self.__messagingSenderId="507874565300"
        self.__appId="1:507874565300:web:696296c125fcbf383e0be1"
        self.__measurementId="G-B95Y9TZL9D"
        self.__databaseURL="https://genescope-c9328-default-rtdb.firebaseio.com/"
        self.__azure_connectionSTR="DefaultEndpointsProtocol=https;AccountName=genescopestorage;AccountKey=NYJ4pfLtiEoPWP6oHFSr5ADk4NLkEmU4DkvUI7j6wH/I7SeijIAAyfXGsNILt9RErB1bwHQtKhhu+AStOgqXtA==;EndpointSuffix=core.windows.net"
        self.__container_name="datasets"
    
    def get_con_str(self):
        return self.__azure_connectionSTR
    def get_cont_name(self):
        return self.__container_name