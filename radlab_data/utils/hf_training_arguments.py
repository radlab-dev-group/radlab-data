import json

from transformers import HfArgumentParser, TrainingArguments


class HFTrainingArgumentHandler:
    def __init__(
        self,
        out_dir: str = None,
        report_to: list = None,
        tr_arguments: TrainingArguments = None,
    ):
        """
        FT TrainingArgument constructor
        :param out_dir:
        :param report_to:
        :param tr_arguments:
        """
        if tr_arguments is not None:
            self._tr_arguments = tr_arguments
        else:
            self._tr_arguments = self._get_default_training_arguments(
                out_dir=out_dir, report_to=report_to
            )

    def to_dict(self):
        """
        Get dict with training_arguments.
        """
        data = {"training_arguments": self.training_arguments.to_dict()}
        return data

    def save(self, out_filepath: str):
        """
        Store training arguments as json file
        :param out_filepath:
        :return:
        """
        data = self.to_dict()
        with open(out_filepath, "w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def from_config(cls, config_json: str):
        """
        Loads training arguments from json config
        :param config_json:
        :return: object of HFTrainingArgumentHandler
        """
        with open(config_json, mode="r") as file:
            data = json.load(file)
        training_args_dict = data["training_arguments"]
        parser = HfArgumentParser(TrainingArguments)
        (training_args,) = parser.parse_dict(training_args_dict)
        return cls(tr_arguments=training_args)

    @property
    def training_arguments(self):
        return self._tr_arguments

    def _get_default_training_arguments(
        self, out_dir: str = None, report_to: list = None
    ):
        """
        Default training arguments initialization.
        :param out_dir:
        :param report_to:
        :return: object of TrainingArguments
        """
        ta = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=50,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=200,
            weight_decay=0.001,
            load_best_model_at_end=True,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=40,
            disable_tqdm=False,
            logging_first_step=False,
            learning_rate=0.0005,
            fp16=True,
            report_to=report_to,
            save_total_limit=5,
            save_strategy="steps",
            save_steps=40000,
            logging_strategy="steps",
            gradient_accumulation_steps=1,
            eval_accumulation_steps=1,
            optim="adamw_torch_fused",
            fp16_full_eval=True,
        )
        return ta
