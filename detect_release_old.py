def perform_inference(customer_Type,identifier_id, video_id, source_path, dest_path, model_type):
    
    import argparse
    import os
    from os import listdir
    from os.path import join, isfile
    import platform
    import sys
    from pathlib import Path
    import numpy as np
    import time
    import torch
    import logging
    import pathlib # 
    # temp = pathlib.PosixPath
    # pathlib.PosixPath = pathlib.WindowsPath
    from PIL import Image
    import easyocr
    import re
    from collections import Counter

    # swasth webhook service
    from app.services.inspection_webhook_service import webhook_service
    webhook_available = True

    # ROOT = os.getcwd()
    ROOT = os.path.dirname(os.path.abspath(__file__))
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

    from models.common import DetectMultiBackend
    from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
    from utils.general import (LOGGER, set_logging, Profile, check_file, check_img_size, check_imshow,
                               check_requirements,
                               colorstr, cv2, download_file_to_temp, increment_path, non_max_suppression, print_args,
                               scale_boxes, strip_optimizer, xyxy2xywh)
    from utils.plots import Annotator, colors, save_one_box
    from utils.torch_utils import select_device, smart_inference_mode

    #from PIL import Image
    start_time = time.time()
    frame_skip = 3
    #video = cv2.VideoCapture(source_path)  # get vid details
    #fps = video.get(cv2.CAP_PROP_FPS)

    # if (fps < 30):
    #     frame_skip = 3 #th
    #     # same_defect_count = 8  ## same_defect_count is not at all related with the fps


    # elif (fps >= 30):
    #     frame_skip = 3 #th
    #     # same_defect_count = 8 #int(fps/ frame_skip) 
    

    if model_type == 'wrc_water':
        weight_file_path = ROOT / 'runs/train/wrc_water_v9_2.2.0/weights/best_striped.pt'
        data_path        = ROOT / 'data/wrc_stage2.2.0.yaml'
    elif model_type == 'wrc_sewer':
        weight_file_path = ROOT / 'runs/train/wrc_sewer_v9_2.2.0/weights/best_striped.pt' 
        data_path        = ROOT / 'data/wrc_stage2.2.0.yaml'
    elif model_type == 'NASSCO_sewer':
        weight_file_path = ROOT / 'runs/train/wrc_water_v9_2.2.0/weights/best_striped.pt'
        data_path        = ROOT / 'data/nassco_stage3.yaml'
    elif model_type == 'Europe_sewer':
        weight_file_path = ROOT / 'runs/train/wrc_water_v9_2.2.0/weights/best_striped.pt'
        data_path        = ROOT / 'data/europe_stage3.yaml'
    else: 
        weight_file_path = ROOT / 'runs/train/wrc_sewer_v9_2.2.0/weights/best_striped.pt'
        data_path        = ROOT / 'data/wrc_stage2.2.0.yaml'
        
    @smart_inference_mode()
    def run(
            weights = weight_file_path,
            source=source_path,
            data= data_path,
            imgsz=(512,512),
            conf_thres=0.23,
            iou_thres=0.45,
            max_det=1000,
            device='0' if torch.cuda.is_available() else 'cpu',
            view_img=False,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            nosave=False,
            classes=None,
            agnostic_nms=False,
            augment=False,
            visualize=False,
            update=False,
            # project=ROOT / 'runs/detect',
            project=dest_path,  # save results to project/name
            name=identifier_id,
            exist_ok=False,
            line_thickness=3,
            hide_labels=False,
            hide_conf=False,
            half=True if torch.cuda.is_available() else False,
            dnn=False,
            vid_stride=1,
    ):
        source = str(source)
        file_path = Path(source)
        global IMGG, defect_label, defect_frame, fps
    
        save_img = not nosave and not source.endswith('.txt')
        if source.lower().startswith(('http://', 'https://')):
            url_path = source.split('?')[0]
            filename = url_path.split('/')[-1]
            is_file = Path(filename).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        else:
            is_file = (file_path.exists() and file_path.is_file()) or Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        if is_url and is_file:
            source = download_file_to_temp(source, dest_path)  # download source from url to temp dir
            LOGGER.info(f"Returned Source {source}")
            # Send processing notification to other service via webhook
            if webhook_available and video_id:
                webhook_service.notify_status_update(
                    customer_type=customer_Type,
                    video_id=video_id,
                    status="PROCESSING",
                    results={"message": "Starting AI inference processing"}
                )
    
        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    
        savedirr = str(save_dir)
        global path_mid

        path_mid = savedirr + '/mid'
        os.makedirs(path_mid, exist_ok=True)
        

        reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
    
        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        model.names = {
            #0: "Encrustation(DEE,3)",
            0: "Attached Deposits(DE,3)",
            #1: "House_Service_Connection(H,3)",
            1: "Hole(H,3)",
            2: "Joint_Displacement(JD,3)",
            #2: "Joint_Displacement(JDL,4)",
            3: "Surface_damage(S,2)",
            4: "Root(R,3)",
            5: "Settled_Deposits(DE,3)",
            6: "Other Obstacles(OB,3)",
            #6: "Stone(OBB,3)",
            #7: "Partial_Blockage(OBP,3)",
            #8: "Complete Blockage(OBZ,4)",
            7: "Other Obstacles(OB,3)",
            8: "Other Obstacles(OB,3)",
            9: "Fracture(F,3)",
            10: "Infiltration(I,4)",
            #10: "Infiltration(IR,4)",
            11: "Deformed(D,3)",
            12: "Crack(C,1)",
            13: "Joint_Displacement(JD,3)",
            #13: "Joint_Displacement_medium(JDM,3)",
            14: "Broken(B,4)",
            15: "Collapse(XP,5)",
            16: "Attached Deposits(DE,3)",
            #17: "Gravel(DER,3)",
            17: "Settled Deposits(DE,3)",
            18: "Attached_deposit(DE,3)"
        }
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)
        # names1 = {
        #     0: "(DE,3)_(DEE,3)",
        #     1: "(H,3)_(NA)",
        #     2: "(JD,4)_(JDL,4)",
        #     3: "(SW,2)_(NA)",
        #     4: "(R,3)_(NA)",
        #     5: "(DE,3)_(DES,3)",
        #     6: "(OB,3)_(OBB,3)",
        #     7: "(OB,3)_(OBP,3)",
        #     8: "(OB,4)_(OBZ,4)",
        #     9: "(F,4)_(NA)",
        #     10: "(I,4)_(IR,4)",
        #     11: "(D,3)_(NA)",
        #     12: "(C,1)_(NA)",
        #     13: "(JD,3)_(JDM,3)",
        #     14: "(B,4)_(NA)",
        #     15: "(X,5)_(XP,5)",
        #     16: "(DEF,3)_(NA)", #not present
        #     17: "(DE,3)_(DER,3)",
        #     18: "(DE,3)_(DEZ,3)"
        # }

        names1 = {
            0: "(DE-A,3)_(NA)",
            1: "(H,3)_(NA)",
            2: "(JD,3)_(NA)",
            3: "(SW,2)_(NA)",
            4: "(R,3)_(NA)",
            5: "(DE-S,3)_(NA)",
            6: "(OB,3)_(NA)",
            7: "(OB,3)_(NA)",
            8: "(OB,4)_(NA)",
            9: "(F,3)_(NA)",
            10: "(I,4)_(NA)",
            11: "(D,3)_(NA)",
            12: "(C,1)_(NA)",
            13: "(JD,3)_(NA)",
            14: "(B,4)_(NA)",
            15: "(X,5)_(NA)",
            16: "(DE-A,3)_(NA)", #not present
            17: "(DE-S,3)_(NA)",
            18: "(DE-A,3)_(NA)"
        }
        # Dataloader
        bs = 1
        if source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            cap = cv2.VideoCapture(source)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            cap.release()
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs
    
        frame_count = 0
        kk_loop = 0
        
        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
        #logger_output3 = (savedirr + "/" + "total_defects_log.txt")
    
        # Initialize tracking variables
        current_defect = None
        batch_start_frame = 0
        Defect_list_time = []
        Defect_list_dis = []
        IMGG = []
        IMGG_org = []
        defect_label = []
        defect_frame = []
        
        timestamp_file = []
        dist_bot = []
        index_mid = []
        mid_frame = []
        object_count = []
        object_name_count= []
        frame_number = []
        name_defect = []
        frame_list = []
        Mid_Imgg = []
        Mid_Label = []



        #with open(logger_output3, 'w') as file31:
        for path, im, im0s, vid_cap, s in dataset:
                
                orig_image = im0s
                
                if frame_count % frame_skip == 0:
                    kk_loop += 1
    
                    with dt[0]:
                        im = torch.from_numpy(im).to(model.device)
                        im = im.half() if model.fp16 else im.float()
                        im /= 255
                        if len(im.shape) == 3:
                            im = im[None]
    
                    with dt[1]:
                        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                        pred = model(im, augment=augment, visualize=visualize)
                        if isinstance(pred, (list, tuple)):
                            pred = pred[0]
    
                    with dt[2]:
                        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    
                    for i, det in enumerate(pred):
                        og_image = im0s.copy()
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                        
                        p = Path(p)
                        save_path = str(save_dir / p.name)
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                        dist_bot = []
                       
                        if (len(det) or (kk_loop == total_frames)):
                            
                           
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                            num_objects = len(det)  # total detections in the current image
                            labels_detect = []
                            num_defect = []
                            
                            if num_objects == 1:
                                
                                
                                labels_detect = names1[int(det[0][-1])]
                            else:
                                labels_in_image = [names1[int(cls)] for *_, cls in det]
                                labels_detect.append(labels_in_image)

                            # print(f'frame number is: {frame},  no. of object is: {num_objects} and label is: {labels_detect}')
                            num_defect = [frame,num_objects,labels_detect]
                            frame_number.append(num_defect)

                            for *xyxy, cls in reversed(det):
                                if save_img or save_crop or view_img:
                                    c = int(cls)
                                    label = None if hide_labels else names[c]
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                                    im0 = annotator.result()

                                    
                                    if current_defect is None:
                                        current_defect = names[c]
                                        batch_start_frame = frame_count
                                        IMGG.append(im0)
                                        IMGG_org.append(og_image)
                                        defect_label.append(names[c])
                                        # defect_frame.append(frame_count)
    
    
                                    #elif ((names[c] == current_defect) and (kk_loop < total_frames)):
                                    elif ((names[c] == current_defect) and (kk_loop < total_frames) and (len(defect_frame) < 100)):
                                        IMGG.append(im0)
                                        IMGG_org.append(og_image)

                                        defect_label.append(names[c])
                                        # defect_frame.append(frame_count)
                                        frame_list.append(frame)

                                    #elif ((names[c] != current_defect) or (kk_loop == total_frames)):
                                    elif ((names[c] != current_defect) or (kk_loop == total_frames) or (len(defect_frame) < 105)):
                                        
                                        if len(IMGG) > 2: ####if in that range
                                      # if len(IMGG) >= same_defect_count: ####if in that range

                                            
                                            mid_idx = len(IMGG) // 2
                                            mid_frame = IMGG[mid_idx]
                                            mid_frame_org = IMGG_org[mid_idx]


                                            n_frame = frame_list[mid_idx]

                                            frame_list = []
                                            

                                            mid_frame_num = batch_start_frame + mid_idx
                                            time_sec = mid_frame_num / (fps * frame_skip)
                                            mins, secs = divmod(time_sec, 60)
                                            hrs, mins = divmod(mins, 60)
                                            timestamp = f"{int(hrs)}:{int(mins)}:{int(secs)}"
                                            timestamp_file = timestamp.replace(':', '-')
    
                                            # # Convert to PIL and OCR
                                            results = reader.readtext(mid_frame_org, detail=0)
                                            #dist_bot = "0.0 m"
                                            dist_bot = "NA"

                                            for j, text in enumerate(results):
                                                text = text.replace('\xa0', ' ').replace('~', '').replace('O', '0').replace('o', '0').replace(",",".").replace(";",'.').replace("?",".").strip()
                                                #print(f"index is : {j}, text is {text}")
                                                #together_text = re.search(r"(\d+\s*\.?\s*\d*)\s*(m|ft|Miles)\b", text, flags=re.IGNORECASE)
                                                together_text = re.search(r"(\b\d+\s*\.\s*\d+\s*)\s*(m|ft|Miles)\b", text, flags=re.IGNORECASE)
                                                if together_text:
                                                    num = together_text.group(1)  
                                                    unit = together_text.group(2)  
                                                    num1 = re.sub(r"\s*\.\s*", ".", num)
                                            
                                                    dist_bot = f"{num1}-{unit}"
                                                    break
                                                
                                                #together_text1 = re.search(r"(ft|Feet|Miles|Mile)\s*(\d*\s*\.?\s*\d+)\b", text, flags=re.IGNORECASE)
                                                together_text1 = re.search(r"(ft|Feet|Miles|Mile)\s*(\b\d+\s*\.\s*\d+\s*)\b", text, flags=re.IGNORECASE)
                                                if together_text1:
                                                    unit = together_text1.group(1)  
                                                    num = together_text1.group(2)
                                                    num1 = re.sub(r"\s*\.\s*", ".", num)
                                            
                                                    dist_bot = f"{num1}-{unit}"  
                                                    break
                                                
                                                #initial_unit_equal = re.search(r"(Feet|MILES|Miles|miles|Mile|ft)\s*=\s*(\d*\s*\.?\s*\d+)", text, flags=re.IGNORECASE)
                                                initial_unit_equal = re.search(r"(Feet|MILES|Miles|miles|Mile|ft)\s*=\s*(\b\d+\s*\.\s*\d+\s*)", text, flags=re.IGNORECASE)
                                                if initial_unit_equal:
                                                    unit = initial_unit_equal.group(1)  
                                                    num = initial_unit_equal.group(2)  
                                                    num1 = re.sub(r"\s*\.\s*", ".", num)
                                                    dist_bot = f"{num1}-{unit}"  
                                                    break
                                                
                                                #initial_unit_colon = re.search(r"(Feet|Miles|MILES|Mile|ft)\s*:\s*(\d*\s*\.?\s*\d+)", text, flags=re.IGNORECASE)
                                                initial_unit_colon = re.search(r"(Feet|Miles|MILES|Mile|ft)\s*:\s*(\b\d+\s*\.\s*\d+\s*)", text, flags=re.IGNORECASE)
                                                if initial_unit_colon:
                                                    unit = initial_unit_colon.group(1)  
                                                    num = initial_unit_colon.group(2)  
                                                    num1 = re.sub(r"\s*\.\s*", ".", num)
                                            
                                                    dist_bot = f"{num1}-{unit}"  
                                                    break
                                                
                                            
                                                #Merge with next token if it's a unit
                                                if j + 1 < len(results) and re.match(r"^(m|ft|Miles)$", results[j + 1].strip(),flags=re.IGNORECASE):
                                                    #text += results[j + 1].strip()
                                                    text = text + '-' + results[j + 1].strip()
                                                matchp = re.search(r"\b\d+\s*\.\s*\d+\s*(?:m|M|ft|FT|Ft)\b", text)
                                                #matchp = re.search(r'\b\d+(?:\s*\.\s*\d+)?(?:\s*\d+)?\s*(?:m|ft)\b',text, flags=re.IGNORECASE)
                                                if matchp:
                                                    dist_bot = matchp.group()
                                                    
                                                if (j != 0) and re.match(r"^(ft[:=]|Miles[:=])$", results[j - 1].strip(),flags=re.IGNORECASE):
                                                    #text += results[j + 1].strip()
                                                    text = text + '-' + results[j - 1].strip()
                                                matchp = re.search(r"\b\d+\s*\.\s*\d+\s*(?:m|M|ft|FT|Ft)\b", text)
                                                #matchp = re.search(r'\b\d+(?:\s*\.\s*\d+)?(?:\s*\d+)?\s*(?:m|ft)\b',text, flags=re.IGNORECASE)
                                                if matchp:
                                                    dist_bot = matchp.group()
                                                #break



                                            # dist_bot = "0.0 m"
                                            # for j, text in enumerate(results):
                                            #     text = text.replace('\xa0', ' ').replace('~', '').replace('O', '0').replace('o', '0').replace(",",".").replace(";",'.').replace("?",".").strip()
                                                
                                            #     together_text = re.search(r"(\d+\.?\d*)(m|M|ft|FT|Ft)\b", text)
                                            #     if together_text:
                                            #         num = together_text.group(1)  
                                            #         unit = together_text.group(2)  
                                            #         dist_bot = f"{num} {unit}"  
                                            #         break

                                            #     # Merge with next token if it's a unit
                                            #     if j + 1 < len(results) and re.match(r"^(m|M|ft|FT|Ft)$", results[j + 1].strip()):
                                            #         #text += results[j + 1].strip()
                                            #         text = text + ' ' + results[j + 1].strip()
                                            #     matchp = re.search(r"\b\d{1,5}\.\d{1,2}\s*(?:m|M|ft|FT|Ft)\b", text)
                                            #     if matchp:
                                            #         dist_bot = matchp.group()
                                            #         break

                                            cd_ts = current_defect + '_(' + timestamp_file + ')'
                                            # cd_dis = current_defect + '_(' + (dist_bot)+ ')'
                                            ans = []
                                            def contains_float(s):
                                                if s!= 'NA':
                                                    for part in s.split("-"):
                                                        try:
                                                            float(part)
                                                            ans = part
                                                            print(ans)
                                                            #dis_num = part
                                                            #print(part)
                                                            #return True  # If any part can be converted to float
                                                        except ValueError:
                                                            continue
                                                            
                                                else:
                                                    ans = 'NA'
                                                return ans
                                            dis_num = contains_float(dist_bot)
                                            #cd_dis = current_defect + '_(' + (dis_num)+ ')'
                                            cd_dis = current_defect + '_(' + (dist_bot) + ')'
                                            #new_dis = float(dis_num[0])
                                            # multiple mid image 
                                            #if ((cd_ts not in Defect_list_time) and (cd_dis not in Defect_list_dis) and (dis_num != 0)) or ((cd_ts not in Defect_list_time) and (dis_num == 0)):
                                            if ((cd_ts not in Defect_list_time) and (cd_dis not in Defect_list_dis) and (dis_num != 0)) or ((cd_ts not in Defect_list_time) and (dis_num == 'NA')):

                                            # multiple mid image 
                                            #if ((cd_ts not in Defect_list_time) and (cd_dis not in Defect_list_dis)):
                                                Defect_list_time.append(cd_ts)
                                                Defect_list_dis.append(cd_dis)
                                                
                                                for rows in frame_number:
                                                        if n_frame in rows:
                                                            #print(f"rows is:{rows}")
                                                            num_def = rows[1]
                                                            if (num_def == 1):
                                                                curr_def = rows[2]
                                                                # text_mid = '{curr_def}_({timestamp_file})_({dist_bot})_(L).png'
                                                                # Mid_Imgg.append(mid_frame)
                                                                # Mid_Label.append(text_mid)
                                                                cv2.imwrite(join(path_mid, f" {curr_def}_({timestamp_file})_({dist_bot})_(L).png"),mid_frame)
                                                                # cv2.imwrite(join(path_mid, f"{curr_def}_({timestamp_file})_({dist_bot})_(WL).png"),mid_frame_org)
    
                                                            elif(num_def >1):
                                                                for index_mid in range(0,num_def):
                                                                    curr_def = rows[2][0][index_mid]
                                                                    # text_mid = '{curr_def}_({timestamp_file})_({dist_bot})_(L)_({index_mid}).png'
                                                                    # Mid_Imgg.append(mid_frame)
                                                                    # Mid_Label.append(text_mid)
                                                                    cv2.imwrite(join(path_mid, f"{curr_def}_({timestamp_file})_({dist_bot})_(L)_({index_mid}).png"),mid_frame)
                                                                    # cv2.imwrite(join(path_mid, f"{curr_def}_({timestamp_file})_({dist_bot})_(WL)_({index_mid}).png"),mid_frame_org)
                                                                
                                                   

                                                
                                                
                                        # Reset for new defect type
                                        # IMGG = [im0]  # Start new batch with current frame _ labeled image
                                        # IMGG_org = [im0]  # Start new batch with current frame _ original image
                                        # defect_label.append(names[c])
                                        # defect_frame.append(frame_count)
                                        # print('defect_label is :\n', defect_label)
                                        # print('defect_frame is :\n', defect_frame)
                                        # print('length is : \n',len(defect_frame))
                                        IMGG = []
                                        IMGG_org = []
                                        defect_label = []
                                        # defect_frame = []
                                        current_defect = names[c]
                                        frame_list = []
                                        batch_start_frame = frame_count
                                        
                            

                                        
                            
                        if save_img:
                            if vid_path[i] != save_path:
                                vid_path[i] = save_path
                                if isinstance(vid_writer[i], cv2.VideoWriter):
                                    vid_writer[i].release()
                                if vid_cap:
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path = str(Path(save_path).with_suffix('.mp4'))
                                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
                            vid_writer[i].write(im0)
    
                    # print(object_count)
                                
                    # print(object_name_count)
                frame_count += 1
                
        # removing the image labels if any one distance is NA
        files = os.listdir(path_mid)
        #print(f"length of mid folder is: {len(files)}")

        # Sort files (optional, if you want to rename in order)
        #files.sort()
        
        Dist_store = []
        unit_store = []
        
        
        # Loop through and rename
        for index, filename in enumerate(files):
            file_split = filename.split('_')
            if (len(file_split) == 5):
                dis = file_split[-2]
                unit_split = dis.split()
                unit_dis = unit_split[-1]
            #else:
            if (len(file_split) == 6):
                dis = file_split[-3]
                unit_split = dis.split()
                unit_dis = unit_split[-1]
            Dist_store.append(dis)
            unit_store.append(unit_dis)
            #print(filename)
        
        # print(Dist_store)
        # print(unit_store)
        # print(f"length of dist folder is: {len(Dist_store)}")
        # print(f"length of unit folder is: {len(unit_store)}")
        # print(len(set(unit_store)))
        #print('(NA)' in Dist_store")
        
        
        if((len(set(unit_store)) > 1) or ('(NA)' in Dist_store) or ('NA' in Dist_store)):
            
            for index, filename in enumerate(files):
                
                file_split = filename.split('_')
                #print(f'file split is {file_split} and  length of file_split is: {len(file_split)}\n')
                new_filename = []
                if (len(file_split) == 5):
                    file_split[-2] = '(NA)'
                    new_filename = f"{file_split[0]}_{file_split[1]}_{file_split[2]}_{file_split[3]}_{file_split[4]}"
                #else:
                if (len(file_split) == 6):
                    file_split[-3] = '(NA)'
                    new_filename = f"{file_split[0]}_{file_split[1]}_{file_split[2]}_{file_split[3]}_{file_split[4]}_{file_split[5]}"
                #if os.path.exists(filename):
                    # os.rename(filename, new_filename
                os.rename(os.path.join(path_mid , filename), os.path.join(path_mid , new_filename))
                # print(f"Renamed: {filename} --> {new_filename}\n")
        
                
           
        end_time = time.time()
        Total_time = end_time - start_time
        print(f"Total processing time: {Total_time/60:.2f} minutes")
    
    def parse_opt():
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=weight_file_path, help='model path or triton URL')
        parser.add_argument('--source', type=str, default=source_path, help='file/dir/URL/glob/screen/0(webcam)')
        parser.add_argument('--data', type=str, default= data_path, help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.23, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=dest_path, help='save results to project/name')
        parser.add_argument('--name', default=identifier_id, help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
        print_args(vars(opt))
        return opt

    def main(opt):
        run(**vars(opt))

    opt = parse_opt()
    main(opt)
    # ## to remove duplicates based on name+dist or name+time
    print(f"Inference Completed for video id {video_id}")

###############################################################################
# from distance_monitor import perform_distance_test
identifier_id="Detect_give_78mb_115319"
source_path = "/home/ec2-user/SageMaker/nvme-storage/videos/115319.mp4"
dest_path = "/home/ec2-user/SageMaker/nvme-storage/test"
model_type = 'wrc_sewer'
perform_inference('',identifier_id,'',source_path,dest_path,model_type)
# # global start_frame
# # start_frame=perform_distance_test(source_path)
# # if start_frame is not None:
# #     perform_inference(identifier_id,'',source_path,dest_path,model_type)
# # else:
# #     print("The video is outside the pipeline")
# perform_inference('',identifier_id,'',source_path,dest_path,model_type)