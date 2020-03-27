import config
class BERTDataset:
    def __init__(self,review, target):
        self.review = review
        self.target = target
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review)
        review = " ".join(review.split())
        inputs = self.tokenizer.encoder_plus(
            review,
            None,
            add_special_tokens=True,
            max_length = self.max_len
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        # for bert we pad on right
        padding_length = self.max_len = len(ids)
        ids = ids + ([0]*padding_length)
        mask = mask + ([0]*padding_length)
        token_type_ids = token_type_ids + ([0]*padding_length)

        return