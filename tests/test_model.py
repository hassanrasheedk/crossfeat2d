import torch

from crossfeat2d.model import ContrastiveLoss, EmbeddingNet, SiameseNetwork, mine_hard_negatives


def test_embedding_shape():
    net = EmbeddingNet()
    assert net(torch.rand(5, 3, 64, 64)).shape == (5, 128)


def test_state_dict_keys_match_notebook_checkpoints():
    # Checkpoints saved by the notebooks are state dicts of SiameseNetwork(UNet()).
    keys = set(SiameseNetwork().state_dict())
    expected = {
        f"embedding_net.{layer}.{p}"
        for layer in ("enc_conv1", "enc_conv2", "enc_conv3", "enc_conv4", "fc1")
        for p in ("weight", "bias")
    }
    assert keys == expected


def test_contrastive_loss_values():
    loss = ContrastiveLoss(margin=1.0)
    a = torch.zeros(2, 4)
    b = torch.tensor([[0.6, 0, 0, 0], [0.6, 0, 0, 0]])
    target = torch.tensor([1.0, 0.0])
    # positive: 0.5 * 0.36; negative: 0.5 * (1 - 0.6)^2
    per_pair = loss(a, b, target, size_average=False)
    assert torch.isclose(per_pair, torch.tensor(0.5 * 0.36 + 0.5 * 0.16))


def test_hard_negative_replaces_easier_negative():
    loss = ContrastiveLoss(margin=1.0)
    a = torch.zeros(1, 2)
    b = torch.tensor([[0.9, 0.0]])  # own negative, far
    hard = torch.tensor([[0.1, 0.0]])  # mined negative, closer
    target = torch.tensor([0.0])
    assert torch.isclose(loss(a, b, target, hard), torch.tensor(0.5 * 0.9**2))


def test_mine_hard_negatives_picks_closest_other():
    anchors = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
    others = torch.tensor([[3.0, 0.0], [0.5, 0.0]])
    target = torch.tensor([0.0, 0.0])
    hard = mine_hard_negatives(anchors, others, target)
    assert torch.equal(hard[0], others[1])


def test_training_step_reduces_loss():
    from torch.utils.data import DataLoader

    from crossfeat2d.data import SiamesePatchDataset
    from crossfeat2d.train import train

    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    pairs, labels = [], []
    for _ in range(32):
        a = torch.rand(64, 64, 3, generator=gen).numpy()
        pos = (a * 0.9).astype("float32")
        neg = torch.rand(64, 64, 3, generator=gen).numpy()
        pairs += [(a, pos), (a, neg)]
        labels += [1, 0]
    loader = DataLoader(SiamesePatchDataset(pairs, labels), batch_size=16, shuffle=True)
    history = train(SiameseNetwork(), loader, loader, epochs=3, log=lambda _: None)
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert "val_loss" in history[0]
