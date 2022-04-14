raw_data_path <- "/home/shared_data/dash/raw_data/dataset7_35Mbps_max_brate_withCa/"
horz_wind_size <- 2
aggr_wind_size <- 12
pred_data_path <- paste0('../data/data_eval_output/dataset7-', 
                      horz_wind_size, 'sWsize-', 
                      aggr_wind_size, 'aggsize/')

runs <- c(6,12,21,27)
wind_size <- 4 # seconds
sim_time <- 1000 # seconds
model_type <- 'lstm'

for (r in runs){
    print(paste("Run: ", r))
    real_dash_log <- paste0("run",r,"/dash_client_log.txt")
    pred_log <- paste0("run",r,"wsize",horz_wind_size,"eval_out_",model_type,".csv")
    mob_log <- paste0("run",r,"/mobility_trace.txt")
    out_fname <- paste0("ground_truth_",pred_log)

    num_clients_file <- paste0(raw_data_path,"run",r,"/","parameter_settings.txt")
    num_clients <- readLines(num_clients_file, n = 5)
    num_clients <- strsplit(num_clients[5], " = ")
    num_clients <- strtoi(num_clients[[1]][2])

    out_file <- file(paste0(pred_data_path, out_fname), "w")
    writeLines(paste(c("Treal_start", "Treal_end", "Node", "video-Id", "seg-Id", "seg-qual", "gNB-Id"), collapse = "\t"), out_file)

    skip_lines <- (if (r==1) 5 else 2)
    real_dash_data <- read.table(paste0(raw_data_path, real_dash_log), header = TRUE, skip = skip_lines, sep ="\t", stringsAsFactors = FALSE,
                         nrow = (length(readLines(paste0(raw_data_path, real_dash_log))) - num_clients - skip_lines - 1))
    real_assoc_data <- read.table(paste0(raw_data_path, mob_log), header = TRUE, sep ="\t", stringsAsFactors = FALSE)

    # add the time_to_nxt_request to the real time line
    real_dash_data$tstamp_us = (real_dash_data$tstamp_us/1000000) + real_dash_data$delayToNxtReq_s 
    real_assoc_data$tstamp_us = (real_assoc_data$tstamp_us/1000000) 

    # Create time windows by using which 
    start_time <- 0 # seconds   
    end_time <- sim_time # seconds
    wind_start <- start_time
    wind_end <- start_time + horz_wind_size
    wind_ind <- 1
    num_wind <- floor( (end_time - start_time) /horz_wind_size)
    num_seg <- array(data = 0, dim = c(num_clients,num_wind)) 
    ground_truth_num_seg <- array(data = 0, dim = c(num_clients,num_wind)) 
    ground_truth_wind_time <- array(data = 0, dim = c(num_clients,num_wind)) 

    while (wind_end <= end_time){
      # get all the samples for this window
      this_wind_real <- real_dash_data[which ((real_dash_data$tstamp_us >= wind_start) & (real_dash_data$tstamp_us < wind_end) ),]
      this_wind_assoc_real <- real_assoc_data[which ((real_assoc_data$tstamp_us >= wind_start) & (real_assoc_data$tstamp_us < wind_end) ),]
      # find all the nodes that exist in this window
      nodes <- sort(unique(this_wind_real[,2]))
      # get the segment qualities requested in this window, the new_brate for each node
      for(node in nodes){
        seg_qual <- this_wind_real[which(this_wind_real[,2] == node),5]
        # the +1 here is because the segment id is logged after a segment is received. 
        # After we add the timeForNxtRequest to the timestamps, we have the time at which
        # a ground truth request was actually sent out. This will be sent out with seg numn = last +1
        seg_id <- this_wind_real[which(this_wind_real[,2] == node),4] + 1 
        video_id <- max(this_wind_real[which(this_wind_real[,2] == node),3])
        assoc <- this_wind_assoc_real[which(this_wind_assoc_real[,3] == node),4] # 4 for assoc celID
        assoc_time <- this_wind_assoc_real[which(this_wind_assoc_real[,3] == node),1]
        seg_time <- this_wind_real[which(this_wind_real[,2] == node),1]
        # add this to file
        for (i in 1:length(seg_id)){
          # get the association for this seg_id
          if(length(assoc_time) == 0 ){
            break
          }
            for (j in 1:length(assoc_time)){
              if(assoc_time[j] >= seg_time[i]){
                assoc_val <- assoc[j]
                break
              }
            }
          w <- c(as.character(wind_start), as.character(wind_end), as.character(node), as.character(video_id), 
                 as.character(seg_id[i]), as.character(seg_qual[i]), as.character(assoc_val))
          writeLines(paste(w, collapse = "\t"), out_file)
        }
      }
      # wrap up this window and set the nexct window limits
      wind_start <- wind_end
      wind_end <- wind_start + horz_wind_size
      wind_ind <- wind_ind + 1
    } # end of while 
} # end of for over runs
