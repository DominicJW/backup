from io import BytesIO
from flask import send_file, url_for
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import zipfile
import torch
from transformers import AutoTokenizer
from model_client import model

import time



tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

app = Flask(__name__)



html = ""
j = 0
pkv = None
attentions = None #interesting concatenation procedure
existing_text = ""


def get_outputs(text):
    toks = tokenizer(text,return_tensors = "pt")["input_ids"]
    outputs = model(input_ids=toks,use_cache=True,output_attentions=True)
    return outputs



@app.route('/', methods=['GET'])
def index():
    global html
    return render_template('index.html', token_html=html)


# Example function that returns a matplotlib Figure for a given token index
def make_token_heatmap_fig(query_token_ind,key_token_ind, attn,width=1, height=1):
    z = torch.concat([layer[:,:,query_token_ind,key_token_ind].to(torch.float64) for layer in attn],axis = 0) #concatenates the attentions
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    im = ax.imshow(z, aspect='auto', cmap='Blues')
    ax.axis('off')
    plt.tight_layout(pad=0)
    if query_token_ind == key_token_ind:
        pass
        # plt.colorbar(im, ax=ax)

    return fig


@app.route('/generate_plots_and_return', methods=['POST'])
def generate_plots_and_return():
    data = request.json
    query_token_ind = data.get('token_index', 0)
    num_toks = j
    print(existing_text)
    # could use tokens as the input

    attn = get_outputs(existing_text)["attentions"] #doesnt use the pkv cache
    #32 layers, 32 Q heads,
    #32,1,32,seqlen(past + new),seqlen (new) # since pkv is not in use, its justs 32,32,1,new,new

    # Create a BytesIO buffer for the ZIP file
    zip_buffer = BytesIO()

    # Create a ZIP file in the BytesIO buffer
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for key_token_ind in range(num_toks):
            fig = make_token_heatmap_fig(query_token_ind,key_token_ind,attn)  # Your function to create the figure
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)
            
            # Add the PNG file to the ZIP file
            zip_file.writestr(f'token_{key_token_ind}.png', buf.getvalue())

    # Seek to the beginning of the ZIP file
    zip_buffer.seek(0)

    return send_file(zip_buffer, mimetype='application/zip',
                     as_attachment=True,
                     download_name=f'tokens_{query_token_ind}.zip')





# Tokenize route that returns HTML with spans and image tags
@app.route('/tokenize', methods=['POST'])
def tokenize():
    global html, j,existing_text
    print("hello")
    data = request.json
    text = data.get('input_text', '')
    

    #could make tree root actually and then we add text/branches by using context menu
    existing_text += text
    tokens = tokenizer.tokenize(text)
    
    pieces = []
    for i, token in enumerate(tokens):
        idx = j + i
        token = token.replace("<","").replace(">","")
        pieces.append(f'<span class="token" data-index="{idx}" oncontextmenu="showContextMenu(event)">{token}'
                      f' <img class="token-heatmap" alt="heatmap" loading="lazy" width="160"></span>')
    j += len(tokens)
    html += "".join(pieces)
    return index()


@app.route("/create_new_branch",methods=["POST"])
def create_new_branch():
    data = request.json
    text = data.get('input_text', '')
    index_string = data.get('token_index',"0")
    index = map(lambda x: int(x),index_string.split(".")) #cast list of str to list of int
    if index_string.contains("."): 
        last_ind = int(index_string[index_string.rfind("."):]) + 1
        new_index = index_string[:index_string.rfind(".")] + str(last_ind)
    else:
        last_ind = int(index_string) + 1
        new_index = str(last_ind)
    
    node = tree.get_item_by_index(index)
    tokens = node.add_text(text) #returns tokens as strings, sends off to model for calcs
    #unfortunatly the html does store the data. 

    end = ""
    html = ""
    for idx,token in enumerate(tokens):
        html += f'<span class="token" data-index={new_index} oncontextmenu="showContextMenu(event)">{token}<img class="token-heatmap" alt="heatmap" loading="lazy" width="160">'
        end += "</span>"
        new_index += ".0"
    html += end
    return html


@app.route('/clear', methods=['GET'])
def clear():
    global html, j,existing_text
    html = ""
    existing_text = ""
    j = 0
    return index()

if __name__ == '__main__':
    app.run(debug=True,use_reloader=True)

#TODO: make sure we are indexing the query and key tokens correctly
#make datastructure for pkv cache and attentions

#make scale more clear
#implement the branching
#improve the interface style (keep it minimal though)
    #be able to pull up an attention square between two tokens, and pin it.
    #and pin multiple for visual comparison

#I need to incorporate the ROPE encodings into the visualisation as they have a big effect. particularly on the first token
#Could do it so we can compare between two attention images
#also i would like to view the attention scores from after. 
#I mean the attention of future tokens 32Queries of those future tokens
#with the current token
#interesting tree traversal



#when the actual token itself is important, you will see higher attention scores on the lower layers, 
#possibly we would see higher attention scores between the token and itself. 

#first token seems to have fully equal attention with itself, which is strange. 


#on front end and backend, need an empty root of the tree