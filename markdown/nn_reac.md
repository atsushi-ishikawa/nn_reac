---
theme: unicorn
---

## Main
```python
# load data and put it to DataFrame
df = pd.read_json(reac_json)

# parameters
numuse = int(numdata * 1.0)
nclass = 5
num_epoch  = 1000
nchannel = 64

# elements
elements = ["Ru", "Pt"]

printnum = 200
batch_size_percent = 20
batch_size = int(batch_size_percent*numdata/100)
z_dim = 100
lr = 1.0e-3
b1 = 0.5
b2 = 0.999
dropoutrate = 0.3
```

---

```python
# divide into groups according to score
rank = pd.qcut(df["score"], nclass, labels=False)
df["rank"] = rank

# Extract atomic numbers to replace, by removing atoms to fix.
df["atomic_numbers_replace"] = df["atomic_numbers"].apply(make_replace_atom_list, fix_element_number=fix_element_number)

criterion = nn.MSELoss()

# define model and optimizer
D = Discriminator().to(device)
G = Generator().to(device)
D_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(b1, b2))
G_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(b1, b2))

# training
for epoch in range(num_epoch):
    D_loss, G_loss = train(D, G, criterion, D_opt, G_opt, dataloader)
    history["D_loss"].append(D_loss)
    history["G_loss"].append(G_loss)

# generate fake samples with GAN
fakesample = []
for target in range(nclass):
    fakesample.append(generate(G, target=target))
```

---

## Discriminator
```python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear((1 + nclass)*natom, 2*nchannel),
            nn.BatchNorm1d(2*nchannel),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropoutrate),

            nn.Linear(2*nchannel, 2*nchannel),
            nn.BatchNorm1d(2 * nchannel),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropoutrate),

            nn.Linear(2*nchannel, 1),

            nn.Sigmoid(),
        )

    def forward(self, input):
        x = input
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
```

---

## Generator
```python
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        n_feature = z_dim * (nclass + 1)
        self.conv = nn.Sequential(
            nn.Linear(n_feature, 2*n_feature),
            nn.BatchNorm1d(2*n_feature),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropoutrate),

            nn.Linear(2*n_feature, 2*n_feature),
            nn.BatchNorm1d(2*n_feature),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropoutrate),

            nn.Linear(2*n_feature, natom),
            nn.Sigmoid()  # output as (0,1)
        )

    def forward(self, input):
        input = input.view(batch_size, -1)
        x = self.conv(input)
        x = x.view(batch_size, -1, 1)
        return x
```

---

## Training
```python
def train(D, G, criterion, D_opt, G_opt, dataloader):
    D.train()
    G.train()

    y_real = torch.ones(batch_size, 1, device=device)
    y_fake = torch.zeros(batch_size, 1, device=device)

    D_running_loss = 0.0
    G_running_loss = 0.0

    for batch_idx, (real_system, label) in enumerate(dataloader):
        z = torch.randn(batch_size, z_dim, 1, device=device)  # randn is better than rand
        
        # updating discriminator
        D_opt.zero_grad()
        D_real = D(real_system_label)
        D_real_loss = criterion(D_real, y_real)

        z_label = concat_vector_label(z, label, nclass, device)
        fake_system = G(z_label)
        fake_system_label = concat_vector_label(fake_system, label, nclass, device)
        D_fake = D(fake_system_label.detach())
        D_fake_loss = criterion(D_fake, y_fake)
```

---

```python
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        D_opt.step()
        D_running_loss += D_loss.item()
        
        # updating Generator
        z = torch.randn(batch_size, z_dim, 1, device=device)
        z_label = concat_vector_label(z, label, nclass, device)
        G_opt.zero_grad()
        fake_system = G(z_label)
        fake_system_label = concat_vector_label(fake_system, label, nclass, device)
        D_fake = D(fake_system_label)
        G_loss = criterion(D_fake, y_real)

        G_loss.backward()
        G_opt.step()
        G_running_loss += G_loss.item()

    return D_running_loss, G_running_loss
```

---

## Predict
```python
def generate(G, target=0):
    scaler = MinMaxScaler()
    G.eval()
    z = torch.randn(batch_size, z_dim, 1, device=device)
    z_label = concat_vector_label(z, target, nclass, device)
    fake = G(z_label)
    fake = fake.detach().cpu().numpy()
    fake = scaler.inverse_transform(fake)
    
    return fake
```
