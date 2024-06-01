import os
import cv2
import re

BIGMAX = (2**53-1)

class VideoQueueManager:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "path":  ("STRING", {"default": '', "multiline": False}),  
                "batch_size": ("INT", {"default": 1, "min": 1}),
                "string_decimals":  ("INT", {"default": 4, "min":0, "max":16}),  
            },
        }

    RETURN_TYPES = ("STRING","STRING","INT","STRING","INT")
    RETURN_NAMES = ("full_path","filename","current_frame","current_frame_string","total_frames")
    FUNCTION = "exec"

    CATEGORY = "queue"

    

    def exec(self, index, path, batch_size, string_decimals ):     
    
        video_suffixes = ["mp4","avi","mov","wmv","gif","webm","flv","mkv","mpg","avchd"]    
            
        file_list = sorted(os.listdir(path), key=lambda s: sum(((s, int(n)) for s, n in re.findall(r'(\D+)(\d+)', 'a%s0' % s)), ()))
        cindex = 0
        for filename in file_list:
            if filename.split(".")[-1].lower() in video_suffixes:
                video_cap = cv2.VideoCapture(path+filename)
                if video_cap.isOpened():
                    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))    
                    video_cap.release()
                    
                    if (cindex+total_frames)/batch_size>index:
                        index-=cindex
                        return (path+os.sep+filename,filename,index,str(index).zfill(string_decimals),total_frames ) 
                    cindex += total_frames    
                else:
                    print("could not open",path+os.sep+filename)
                        
        return ("","",-1)        
        
class FolderQueueManager:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": BIGMAX }),
                "path":  ("STRING", {"default": '', "multiline": False}),  
                
            },
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("full_path","foldername")
    FUNCTION = "exec"

    CATEGORY = "queue"

    def exec(self, index, path ):     
        
        file_list = sorted(os.listdir(path), key=lambda s: sum(((s, int(n)) for s, n in re.findall(r'(\D+)(\d+)', 'a%s0' % s)), ()))
        cindex = 0
        for filename in file_list:
            if os.path.isdir(path+os.sep+filename):
                index-=1
                if index==-1:
                    return(path+os.sep+filename,filename)
               
        return ("","")               
    
