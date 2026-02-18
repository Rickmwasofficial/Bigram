# A simple 4-gram model trained to generate swahili text
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.theme import Theme
from rich import print as pprint
import time
import os
import torch

custom_theme = Theme({
    "info":    "cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error":   "bold red",
    "debug":   "dim white",
})

console = Console(theme=custom_theme)


with console.status("[cyan]Loading training data...", spinner="dots"):
    # Data loading
    data = open("../data/train.txt").read()
    train_text = data.replace("UNK", '').replace('\n', '')
    console.print(f"[success] SUCCESS[/] Loaded {len(train_text):,} characters!")

# tokenizing the text
stoi = {s:i for i, s in enumerate(sorted(list(set(train_text))))}
itos = {i:s for i, s in enumerate(sorted(list(set(train_text))))}

# Create the 4-gram lookup table
B = torch.ones((27, 27, 27, 27), dtype=torch.int32)

with Progress(
    SpinnerColumn(spinner_name="dots2"),
    TextColumn("[cyan]{task.description}"),
    TimeElapsedColumn(),
    console=console
) as progress:
    task = progress.add_task("Checking and loading saved model...", total=None)
      # total=None = indeterminate
    if os.path.exists("./model.pt"):
        B = torch.load("./model.pt")
        progress.update(task, description="[success] Loading saved model...done[/]", completed=1, total=1)
       
    else:
        progress.update(task, description="[info] No saved model... Prceeding[/]", completed=1, total=3)
       
        task1 = progress.add_task("Training 4-gram model...", total=None)
        for ch1, ch2, ch3, ch4 in zip(train_text, train_text[1:], train_text[2:], train_text[3:]):
            id1, id2, id3, id4 = stoi[ch1], stoi[ch2], stoi[ch3], stoi[ch4]
            B[id1, id2, id3, id4] += 1
        progress.update(task1, description="[success] Training bigram model...done[/]", completed=2, total=3)
        task2 = progress.add_task("Saving trained model...", total=None)
        
        torch.save(B, "model.pt")
        progress.update(task2, description="[success] Saving trained model...done[/]", completed=3, total=3)
       


def generate_text(start: str, length = 500):
    g = torch.Generator().manual_seed(12482828)
    count = 0
    ix, ix2, ix3 = stoi[start[0]], stoi[start[1]], stoi[start[2]]
    P = B.float()
    P = P / P.sum(3, keepdim=True)
    temperature = 1.1
    print(start, end='', flush=True)  # print the seed characters first

    while True:
        k = 8
        p = P[ix, ix2, ix3].clone()
        p = p ** (1 / temperature)
        p = p / p.sum()
        top_k_vals, top_k_idx = torch.topk(p, k)
        mask = torch.zeros_like(p)
        mask[top_k_idx] = 1
        p = p * mask
        p = p / p.sum()
        out = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        ix, ix2, ix3 = ix2, ix3, out
        print(itos[out], end='', flush=True)
        time.sleep(0.05)
        count += 1
        if count == length:
            break

console.print(f"[info] INFO[/] Starting text generation...\n")
generate_text(start="ana")


