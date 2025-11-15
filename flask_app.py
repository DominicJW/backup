from io import BytesIO
from flask import send_file, url_for
import matplotlib
from matplotlib import cm
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
from TextTree import TextTree
import time



tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

app = Flask(__name__)

#

html = ""
j = 0
existing_text = ""

tree = TextTree(model = model, tokenizer = tokenizer)

def get_outputs(text):
    toks = tokenizer(text,return_tensors = "pt")["input_ids"]
    outputs = model(input_ids=toks,use_cache=True,output_attentions=True)
    return outputs



@app.route('/', methods=['GET'])
def index():
    global html
    return render_template('index.html', token_html=html)



def make_token_heatmap_fig_colormap_groups(attn_input, col_groups, cmaps, width=4, height=2):
    rows, cols = attn_input.shape
    vmin, vmax = attn_input.min(), attn_input.max()
    norm = (attn_input - vmin) / (vmax - vmin + 1e-12)

    rgb = np.zeros((rows, cols, 3))
    for (start, end), cmap_name in zip(col_groups, cmaps):
        cmap = cm.get_cmap(cmap_name)
        seg = norm[:, start:end]
        # map each value to RGBA, drop A
        rgb[:, start:end, :] = cmap(seg)[:, :, :3]

    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    ax.imshow(rgb, aspect='auto')
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig



def make_token_heatmap_fig(attn_input,width=1, height=1):    
    #how to make this put different colours for different columns?
    #key being for the four groups of Query heads
    groups = [(0,8),(8,16),(16,24),(24,32)]
    cmaps = ['Reds','Blues','Greens','Purples']
    fig = make_token_heatmap_fig_colormap_groups(attn_input, groups, cmaps, width=1, height=1)
    return fig


@app.route('/generate_plots_and_return', methods=['POST'])
def generate_plots_and_return():
    #TODO: in future we should traverse later branches
    #we can traverse all children to find the all leaves of this node


    #Also would be good to view the tokens attention with itself as a different colour
    #
    data = request.json
    query_token_ind = data.get('token_index', None)

    index_string = data.get('token_index',"")
    index = list(map(lambda x: int(x),filter(lambda x: x != "",index_string.split(".")))) 
    selected_node = tree.get_item_by_index(index)
    attn = selected_node.attentions
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        token_index_in_attention = len(index)
        for i in range(len(index)):
            other_token_ind = ".".join(map(lambda x: str(x),index[:i+1]))
            attn_input = torch.concat([layer[:,:,token_index_in_attention-1,i].to(torch.float64) for layer in attn],axis = 0) #concatenates the attentions #are we certain on the shape?
            buf = BytesIO()
            fig = make_token_heatmap_fig(attn_input = attn_input)
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0) 
            plt.close(fig)
            buf.seek(0)   

            # Add the PNG file to the ZIP file

            zip_file.writestr(f'token_{other_token_ind}.png', buf.getvalue())
        
        for leaf in selected_node.get_leaves():
            nxt_attention = leaf.attentions
            for i in range(token_index_in_attention,nxt_attention[0].shape[2]):
                other_token_ind = ".".join(map(lambda x : str(x), leaf.index[:i+1])) #I think 
                attn_input = torch.concat([layer[:,:,i,token_index_in_attention-1].to(torch.float64) for layer in nxt_attention],axis = 0) #notice key difference to the above version of this
                buf = BytesIO()
                fig = make_token_heatmap_fig(attn_input = attn_input)
                fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0) 
                plt.close(fig)
                buf.seek(0)   
                zip_file.writestr(f'token_{other_token_ind}.png', buf.getvalue())

    # Seek to the beginning of the ZIP file
    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip',
                     as_attachment=True,
                     download_name=f'tokens_{".".join(map(lambda x: str(x),index))}.zip')






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
    #
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

    index_string = data.get('token_index',"")
    index = list(map(lambda x: int(x),filter(lambda x: x != "",index_string.split(".")))) #cast list of str to list of int #can be
    node = tree.get_item_by_index(index) #empty index means return the root #in the textreee code when it gets to empty index it returns the object as is
    tokens,new_index_single = node.add_text(text) #returns tokens as strings, sends off to model for calcs
    

    end = ""
    html = ""
    if len(index_string) == 0:
        new_index = f"{new_index_single}"
    else:
        new_index = index_string+f".{new_index_single}"

    for idx,token in enumerate(tokens):
        token = token.replace("<","").replace(">","")
        html += f'<span class="token-container" data-index={new_index} oncontextmenu="showContextMenu(event)"> <span class ="token" data-index={new_index}>{token}<img class="token-heatmap" alt="heatmap" loading="lazy" width="160"></span>'
        end += "</span>"
        new_index += ".0"

    html += end
    return html
#

@app.route('/clear', methods=['GET'])
def clear():
    tree.__init__(model = tree.model,tokenizer=tree.tokenizer)
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

#should stop it from allowing two branch values to have same value


#To hide/mask the effect of the positional encodings, we could randomly sample many sentences and subtract the average attention scores from these ones
#average from the branches being displayed that is.


#TODO,
#Have a displa showing which attentions scores are being plotted

#I 
