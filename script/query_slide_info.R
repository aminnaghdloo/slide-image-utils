#2020-04-27
library("RPostgreSQL")
library("stringr")
hostname = "csi-db.usc.edu" # 4db.usc.edu
portnumber = "5432"
databasename = "test_msg"
username = "reader"
pwd = "Meta$ta$i$20!7"

ocular_db_query <- function(query){
  drv <- dbDriver("PostgreSQL")
  con <- dbConnect(drv, host=hostname, port=portnumber, dbname=databasename, 
                   user=username, password=pwd)
  results <- dbSendQuery(con, query )
  analysis_table <- fetch(results, n=-1)
  dbDisconnect(con)
  dbUnloadDriver(drv)
  return(analysis_table)
}

args = commandArgs(trailingOnly=TRUE)
df1 = read.table(args[1], header=TRUE, sep='\t')
slides = paste0(as.vector(df1$slide_id), collapse="','")
query = paste0(
    "select ss.slide_id,ss.tube_id,",
    "tt.patient_id,tt.name as friendly_id,staining_batch_id,",
    "pr.name as staining_protocol,ct.name as cancer_type,",
    "tti.name as tube_type,tt.wbc_count,aa.dapi_count_ocular,",
    "sc.scanner_id,sc.dapi_exposure,sc.tritc_exposure,sc.cy5_gain,",
    "sc.fitc_exposure ",
    "from slide as ss ",
    "join tube as tt on ss.tube_id = tt.tube_id ",
    "join scanners as sc on ss.slide_id = sc.slide_id ",
    "join analysis as aa on ss.slide_id = aa.slide_id ",
    "join protocol as pr on ss.protocol_id = pr.protocol_id ",
    "join cancer_type as ct on tt.cancer_type_id = ct.cancer_type_id ",
    "join tube_type as tti on tti.tube_type_id = tt.tube_type_id ",
    "where ss.slide_id in ('", slides, "')"
)
df2 = ocular_db_query(query)
df3 = merge(df2, df1, 'slide_id', all.y=TRUE)
write.table(df3, args[1], sep='\t', quote=FALSE, row.names=FALSE)
