import pandas as pd
import sys
import os

database_path_pref = "/mnt/N/Amin/reimaging_data_files/" # "/mnt/Y/DZ/"
database_path_suff = "/file_based/data.xml"

if __name__ == "__main__":

    if len(sys.argv) != 3:
        sys.exit(
            f"Two arguments are required: "
            f"python preprocess_for_reimaging.py <input file> <slide_id>"
        )
    else:
        input_path = sys.argv[1]
        slide_id = sys.argv[2]
        slide_path = database_path_pref + slide_id
        output_path = slide_path + database_path_suff
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.isfile(input_path):
        sys.exit(f"Cannot find input file in {input_path}!")
    elif not os.path.isdir(slide_path):
        sys.exit(f"Cannot find slide path in {slide_path}")
    else:
        df = pd.read_table(input_path)
        df.x = df.x.astype("int")
        df.y = df.y.astype("int")
        df["label"] = df.label.astype("int")
        df["forty_x"] = None
        df["picked"] = None
        df["scope"] = None
        df["coord_ix80i_x"] = None
        df["coord_ix80i_y"] = None
        df["coord_zeiss1_x"] = None
        df["coord_zeiss1_y"] = None
        df["coord_olympus_ix81_x"] = None
        df["coord_olympus_ix81_y"] = None
        df["coord_olympus_ix83_x"] = None
        df["coord_olympus_ix83_y"] = None
        df["forty_x"] = None
        df["timestamp"] = None
        df["picking_id"] = None
        df["date_picked"] = None
        df["picked"] = None
        df["unique_id"] = None

        if not "slide_id" in df.columns:
            df.insert(loc=0, column="slide_id", value=slide_id)

        df.to_xml(
            path_or_buffer=output_path,
            index=False,
            root_name="response",
            row_name="cell",
            pretty_print=False,
            parser="etree",
        )
