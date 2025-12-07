#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import shutil
import argparse

import yaml
import torch
from rdkit import Chem

from data.bioparse.writer.complex_to_pdb import complex_to_pdb
from utils.config_utils import overwrite_values
from utils.logger import print_log
from utils.random_seed import setup_seed
import utils.register as R
from models.LDM.data_utils import OverwriteTask, _get_item, Recorder

from .pdb_dataset import PDBDataset, ComplexDesc
from .templates import BaseTemplate


def parse():
    parser = argparse.ArgumentParser(description='Generate peptides given epitopes')
    parser.add_argument('--config', type=str, required=True, help='Path to the test configuration')
    parser.add_argument('--ckpt', type=str, default='./checkpoints/model.ckpt', help='Path to checkpoint')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save generated peptides')

    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 for cpu')
    parser.add_argument('--n_cpus', type=int, default=4, help='Number of CPU to use (for parallelly saving the generated results)')
    return parser.parse_known_args()


# utils
def to_device(data, device):
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
    elif isinstance(data, list) or isinstance(data, tuple):
        res = [to_device(item, device) for item in data]
        data = type(data)(res)
    elif hasattr(data, 'to'):
        data = data.to(device)
    return data


def data_to_cplx(cplx_desc: ComplexDesc, template: BaseTemplate, S: list, X: list, A: list, ll: list, inter_bonds: tuple, intra_bonds: tuple, out_path: str, check_validity: bool=False, check_filters: bool=False):
    '''
        Args:
            bonds: [row, col, prob, type], row and col are atom index, prob has confidence and distance
    '''

    task = OverwriteTask(
        cplx = cplx_desc.cplx,
        select_indexes = cplx_desc.pocket_block_ids + cplx_desc.lig_block_ids,
        generate_mask = cplx_desc.generate_mask,
        target_chain_ids = cplx_desc.tgt_chains,
        ligand_chain_ids = cplx_desc.lig_chains,
        S = S,
        X = X,
        A = A,
        ll = ll,
        inter_bonds = inter_bonds,
        intra_bonds = intra_bonds,
        out_path = out_path
    )

    def template_filter(cplx):
        cplx_desc.cplx = cplx
        return template.validate(cplx_desc)

    cplx, gen_mol, overwrite_indexes = task.get_overwritten_results(
        check_validity = check_validity,
        filters = [template_filter] if check_filters else None
    )

    if cplx is None or gen_mol is None:
        return None, None

    cplx_desc.cplx = cplx

    details = {
        'pmetric': task.get_total_likelihood(),
        'smiles': Chem.MolToSmiles(gen_mol),
        'gen_seq': task.get_generated_seq()
    }

    return cplx_desc, details


def format_log(cplx_desc: ComplexDesc, details: dict, n: int):
    return {
        'id': cplx_desc.id,
        'n': n,
        'pmetric': details['pmetric'],
        'smiles': details['smiles'],
        'gen_seq': details['gen_seq'],
        'tgt_chains': cplx_desc.tgt_chains,
        'lig_chains': cplx_desc.lig_chains,
        'gen_block_idx': cplx_desc.lig_block_ids,
    }


def generate_for_one_template(model, dataset, n_samples, batch_size, save_dir, device, sample_opt):
    recorder = Recorder(dataset, n_samples, save_dir)
    tmp_save_dir = os.path.join(save_dir, 'tmp')
    n_cycles = 1 if dataset.config.name == 'Molecule' else 0    # at least reconstruct once for small molecules to rectify geometry
    
    while not recorder.is_finished():
        batch_list = recorder.get_next_batch_list(batch_size)
        batch = dataset.collate_fn([dataset[i] for i, _ in batch_list])
        
        with torch.no_grad():
            cplx_descs = batch.pop('cplx_desc')
            # to GPU
            batch = to_device(batch, device)
            # inference
            batch_S, batch_X, batch_A, batch_ll, batch_inter_bonds, batch_intra_bonds = model.sample(sample_opt=sample_opt, **batch)

        vae_batch_list = []
        for i in range(len(cplx_descs)):
            S, X, A, ll = batch_S[i], batch_X[i], batch_A[i], batch_ll[i]
            inter_bonds, intra_bonds = batch_inter_bonds[i], batch_intra_bonds[i]
            item_idx, n = batch_list[i]
            cplx_desc = cplx_descs[i]

            if n == 0:
                os.makedirs(os.path.join(tmp_save_dir, cplx_desc.id), exist_ok=True)
                os.makedirs(os.path.join(save_dir, cplx_desc.id), exist_ok=True)
                complex_to_pdb(cplx_desc.cplx, os.path.join(tmp_save_dir, cplx_desc.id, 'pocket.pdb'), cplx_desc.tgt_chains)

            if n_cycles == 0: out_path = os.path.join(save_dir, cplx_desc.id, str(n) + '.pdb')
            else: out_path = os.path.join(tmp_save_dir, cplx_desc.id, str(n) + '.pdb')
            cplx_desc, details = data_to_cplx(
                cplx_desc, dataset.config, S, X, A, ll, inter_bonds, intra_bonds, out_path,
                check_validity = dataset.config.name == 'Molecule' and (n_cycles == 0),
                check_filters = (n_cycles == 0)
            )
            if cplx_desc is None:
                log = None
                assert n_cycles == 0  # only the last cycle has such possibility
            else: log = format_log(cplx_desc, details, n) 
            
            if n_cycles == 0: recorder.check_and_save(log, item_idx, n)
            else:
                data = _get_item(
                    os.path.join(tmp_save_dir, cplx_desc.id, f'pocket.pdb'),
                    out_path.rstrip('.pdb') + '.sdf',
                    cplx_desc.tgt_chains
                )
                data['cplx_desc'] = cplx_desc
                vae_batch_list.append(data)

        model_autoencoder = model.autoencoder
        for cyc_i in range(n_cycles):
            print_log(f'Cycle: {cyc_i}', level='DEBUG')
            final_cycle = cyc_i == n_cycles - 1
            batch = dataset.collate_fn(vae_batch_list)
            vae_batch_list = []
            with torch.no_grad():
                if final_cycle: batch['topo_generate_mask'] = torch.zeros_like(batch['generate_mask'])
                cplx_descs = batch.pop('cplx_desc')
                batch = to_device(batch, device)
                batch_S, batch_X, batch_A, batch_ll, batch_inter_bonds, batch_intra_bonds = model_autoencoder.generate(**batch)
            for i in range(len(cplx_descs)):
                S, X, A, ll = batch_S[i], batch_X[i], batch_A[i], batch_ll[i]
                inter_bonds, intra_bonds = batch_inter_bonds[i], batch_intra_bonds[i]
                item_idx, n = batch_list[i]
                cplx_desc = cplx_descs[i]

                if final_cycle: out_path = os.path.join(save_dir, cplx_desc.id, str(n) + '.pdb')
                else: out_path = os.path.join(tmp_save_dir, cplx_desc.id, f'{n}_cyc{cyc_i}.pdb')
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                cplx_desc, details = data_to_cplx(
                    cplx_desc, dataset.config, S, X, A, ll, inter_bonds, intra_bonds, out_path,
                    check_validity = dataset.config.name == 'Molecule' and final_cycle,
                    check_filters = final_cycle
                )
                if cplx_desc is None:
                    log = None
                    assert final_cycle  # only the last cycle has such possibility
                else: log = format_log(cplx_desc, details, n) 

                if final_cycle: recorder.check_and_save(log, item_idx, n)
                else:
                    data = _get_item(
                        os.path.join(tmp_save_dir, cplx_desc.id, f'pocket.pdb'),
                        out_path.rstrip('.pdb') + '.sdf',
                        cplx_desc.tgt_chains
                    )
                    data['cplx_desc'] = cplx_desc
                    vae_batch_list.append(data)

    print_log(f'Failed rate: {recorder.num_failed / recorder.num_generated}', level='DEBUG')
    shutil.rmtree(tmp_save_dir)


def main(args, opt_args):
    config = yaml.safe_load(open(args.config, 'r'))
    config = overwrite_values(config, opt_args)

    # load model
    ckpt = args.ckpt
    print_log(f'Using checkpoint {ckpt}')
    model = torch.load(ckpt, map_location='cpu', weights_only=False)
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    sample_opt = config.get('sample_opt', {})
    
    # load dataset and dataloader
    batch_size = config.get('batch_size', 32)
    for template in config['templates']:
        print_log(f'Generating for template: {template}')
        out_dir = os.path.join(args.save_dir, template['class'], 'candidates')
        os.makedirs(out_dir, exist_ok=True)
        dataset = PDBDataset(**config['dataset'], template_config=R.construct(template))
        generate_for_one_template(model, dataset, config['n_samples'], batch_size, out_dir, device, sample_opt)


if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    setup_seed(12)
    main(args, opt_args)
