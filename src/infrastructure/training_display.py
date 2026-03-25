from src.infrastructure.configs_general import VERBOSE_BATCHES
from src.infrastructure.dataset_context.dataset_context import DatasetContextAbstract
from typing import List, TYPE_CHECKING
from src.infrastructure.others import round_float, get_custom_model_sparsity_percent
from dataclasses import dataclass

if TYPE_CHECKING:
    from src.infrastructure.layers import LayerComposite

@dataclass
class ArgsTrainingDisplay:
    dataset_context: DatasetContextAbstract
    average_losses_names: List[str]
    model: 'LayerComposite'
    batch_print_rate: int

    get_epoch: any

class TrainingDisplay:
    def __init__(self, args: ArgsTrainingDisplay):
        self.args = args
        self._initalize_losses_array()

    def _initalize_losses_array(self):
        self.losses: List[List[float]]
        self.losses = []
        for loss_name in self.args.average_losses_names:
            self.losses.append([])

    def record_losses(self, losses_per_batch: List[float]):
        for idx, loss in enumerate(losses_per_batch):
            self.losses[idx].append(loss)

        # at print rate losses we print and reinitialize everything
        context: DatasetContextAbstract = self.args.dataset_context
        batch_index = context.get_batch_training_index()
        total_batches = context.get_total_batches_training()
        batch_print_rate = self.args.batch_print_rate

        if batch_index % batch_print_rate == 0 or batch_index == total_batches:
            total_data_len = context.get_data_training_length()
            batch_size = context.get_batch_size()
            model = self.args.model
            epoch = self.args.get_epoch()
            sparsity = get_custom_model_sparsity_percent(model)

            loss_string = ""
            for idx, loss_name in enumerate(self.args.average_losses_names):
                loss_string += loss_name + ": " + str(round_float(sum(self.losses[idx]) / batch_print_rate)) + " | "

            iterated_samples = batch_index * batch_size
            if iterated_samples > total_data_len:
                # This happens in our last batch where we don't iterate all the samples because there are no more left
                iterated_samples = total_data_len

            final_message = f"Train Epoch: {epoch} [{iterated_samples}/{total_data_len}] | Sparsity: {sparsity}"
            if VERBOSE_BATCHES:
                final_message += f"| " + loss_string
            print(final_message)

            self._initalize_losses_array()
