# importing module
import logging

# Create and configure logger
logging.basicConfig(filename="kaif.txt",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)
# s=10
# det=[0,1,2,3,4,5]
# save_dir ='C:/Users/Kaif Ibrahim/Desktop/solinas_downloads'
# Test messages
#logger.debug("Harmless debug Message")
# logger.info(f"{s}{'' if len(det) else '(no detections), '}ms")
# logger.info(f"Results saved to {('bold', save_dir)}{s}")
# logger.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape ')
logger.info("Just an information")
logger.warning("Its a Warning")
logger.error("Did you try to divide by zero")
logger.critical("Internet is down")