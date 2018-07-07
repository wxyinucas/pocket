import random


class Card:
    suit_names = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
    rank_names = [None, 'Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']

    def __init__(self, suit=0, rank=2):

        if suit not in list(range(4)):
            raise RuntimeError('Not legal Suit {} in 0-3'.format(suit))

        if rank not in list(range(1, 14)):
            raise RuntimeError('Not legal Rank {} in 1-13'.format(rank))

        self.suit = suit
        self.rank = rank

    def __str__(self):
        return '{} of {}'.format(Card.rank_names[self.rank], Card.suit_names[self.suit])

    def __lt__(self, other):
        return self.rank < other.rank

    def __eq__(self, other):
        return self.rank == other.rank

    def __le__(self, other):
        return self.rank <= other.rank


class Deck:

    def __init__(self):
        self.cards = [Card(i, j) for i in range(4) for j in range(1, 14)]

    def __str__(self):
        res = [str(card) for card in self.cards]
        return '\n'.join(res)

    def pop_card(self):
        return self.cards.pop()

    def add_card(self, card):
        self.cards.append(card)

    def shuffle(self):
        random.shuffle(self.cards)

    def move_cards(self, hand, num):
        for i in range(num):
            hand.add_card(self.pop_card())


class Hand(Deck):

    def __init__(self, label=''):
        self.cards = []
        self.label = label

