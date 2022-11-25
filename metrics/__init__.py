from metrics import e_recall, nmi, f1, mAP, mAP_c, mAP_1000, mAP_lim, prob_acc
from metrics import logl, unc_cos, unc_l2, unc_logppk
from metrics import dists, rho_spectrum, norms, uncertain_images
from metrics import c_recall, c_nmi, c_f1, c_mAP_c, c_mAP_1000, c_mAP_lim
import numpy as np
import contextlib
import faiss
import torch
from sklearn.preprocessing import normalize
from tqdm import tqdm
import copy
from utilities.misc import log_ppk_vmf_vec


def select(metricname, opt):
    #### Metrics based on euclidean distances
    if 'e_recall' in metricname:
        k = int(metricname.split('@')[-1])
        return e_recall.Metric(k)
    elif metricname=='nmi':
        return nmi.Metric()
    elif metricname=='mAP':
        return mAP.Metric()
    elif metricname=='mAP_c':
        return mAP_c.Metric()
    elif metricname=='mAP_lim':
        return mAP_lim.Metric()
    elif metricname=='mAP_1000':
        return mAP_1000.Metric()
    elif metricname=='f1':
        return f1.Metric()
    elif metricname=='prob_acc':
        return prob_acc.Metric()

    #### Metrics based on cosine similarity
    elif 'c_recall' in metricname:
        k = int(metricname.split('@')[-1])
        return c_recall.Metric(k)
    elif metricname=='c_nmi':
        return c_nmi.Metric()
    elif metricname=='c_mAP':
        return c_mAP.Metric()
    elif metricname=='c_mAP_c':
        return c_mAP_c.Metric()
    elif metricname=='c_mAP_lim':
        return c_mAP_lim.Metric()
    elif metricname=='c_mAP_1000':
        return c_mAP_1000.Metric()
    elif metricname=='c_f1':
        return c_f1.Metric()

    #### calibration metrics
    elif "logl" in metricname:
        return logl.Metric()
    elif "unc_cos" in metricname:
        return unc_cos.Metric()
    elif "unc_l2" in metricname:
        return unc_l2.Metric()
    elif "unc_logppk" in metricname:
        return unc_logppk.Metric()

    #### Generic Embedding space metrics
    elif 'dists' in metricname:
        mode = metricname.split('@')[-1]
        return dists.Metric(mode)
    elif 'rho_spectrum' in metricname:
        mode = int(metricname.split('@')[-1])
        embed_dim = opt.rho_spectrum_embed_dim
        return rho_spectrum.Metric(embed_dim, mode=mode, opt=opt)
    elif 'norms' in metricname:
        quant = int(metricname.split('@')[-1])
        return norms.Metric(quant)
    elif 'uncertain_images' in metricname:
        return uncertain_images.Metric()
    else:
        raise NotImplementedError("Metric {} not available!".format(metricname))




class MetricComputer():
    def __init__(self, metric_names, opt):
        self.pars            = opt
        self.metric_names    = metric_names
        self.list_of_metrics = [select(metricname, opt) for metricname in metric_names]
        self.requires        = [metric.requires for metric in self.list_of_metrics]
        if 'proxyvmf' in opt.loss:
            self.requires.append(["nearest_features_cosine"])
        self.requires        = list(set([x for y in self.requires for x in y]))

    def compute_standard(self, opt, model, dataloader, evaltypes, device, **kwargs):
        evaltypes = copy.deepcopy(evaltypes)

        n_classes = opt.n_classes
        image_paths     = np.array([x[0] for x in dataloader.dataset.image_list])
        _ = model.eval()

        ###
        feature_colls  = {key:[] for key in evaltypes}

        ###
        with torch.no_grad():
            target_labels = []
            final_iter = tqdm(dataloader, desc='Embedding Data...'.format(len(evaltypes)))
            image_paths= [x[0] for x in dataloader.dataset.image_list]
            for idx, inp in enumerate(final_iter):
                context = torch.cuda.amp.autocast() if opt.use_float16 else contextlib.nullcontext()
                input_img,target = inp[1], inp[0]
                if opt.use_float16:
                        input_img = to(torch.float16)

                with context:
                    target_labels.extend(target.numpy().tolist())
                    out = model(input_img.to(device))
                    if isinstance(out, tuple): out, aux_f = out

                    ### Include embeddings of all output features
                    for evaltype in evaltypes:
                        if isinstance(out, dict):
                            feature_colls[evaltype].extend(out[evaltype].cpu().detach().numpy().tolist())
                        else:
                            feature_colls[evaltype].extend(out.cpu().detach().numpy().tolist())


            target_labels = np.hstack(target_labels).reshape(-1,1)


        computed_metrics = {evaltype:{} for evaltype in evaltypes}
        extra_infos      = {evaltype:{} for evaltype in evaltypes}


        ###
        faiss.omp_set_num_threads(self.pars.kernels)
        # faiss.omp_set_num_threads(self.pars.kernels)
        res = None
        torch.cuda.empty_cache()
        if self.pars.evaluate_on_gpu:
            res = faiss.StandardGpuResources()


        import time
        for evaltype in evaltypes:
            features        = np.vstack(feature_colls[evaltype]).astype('float32')
            features_cosine = normalize(features, axis=1)

            start = time.time()

            """============ Compute k-Means ==============="""
            if 'kmeans' in self.requires:
                ### Set CPU Cluster index
                cluster_idx = faiss.IndexFlatL2(features.shape[-1])
                if res is not None: cluster_idx = faiss.index_cpu_to_gpu(res, 0, cluster_idx)
                kmeans            = faiss.Clustering(features.shape[-1], n_classes)
                kmeans.niter = 20
                kmeans.min_points_per_centroid = 1
                kmeans.max_points_per_centroid = 1000000000
                ### Train Kmeans
                kmeans.train(features, cluster_idx)
                centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, features.shape[-1])

            if 'kmeans_cosine' in self.requires:
                ### Set CPU Cluster index
                cluster_idx = faiss.IndexFlatL2(features_cosine.shape[-1])
                if res is not None: cluster_idx = faiss.index_cpu_to_gpu(res, 0, cluster_idx)
                kmeans            = faiss.Clustering(features_cosine.shape[-1], n_classes)
                kmeans.niter = 20
                kmeans.min_points_per_centroid = 1
                kmeans.max_points_per_centroid = 1000000000
                ### Train Kmeans
                kmeans.train(features_cosine, cluster_idx)
                centroids_cosine = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, features_cosine.shape[-1])
                centroids_cosine = normalize(centroids,axis=1) # TODO: Copy paste error? Should be centroids_cosine?


            """============ Compute Cluster Labels ==============="""
            if 'kmeans_nearest' in self.requires:
                faiss_search_index = faiss.IndexFlatL2(centroids.shape[-1])
                if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(centroids)
                _, computed_cluster_labels = faiss_search_index.search(features, 1)

            if 'kmeans_nearest_cosine' in self.requires:
                faiss_search_index = faiss.IndexFlatIP(centroids_cosine.shape[-1])
                if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(centroids_cosine)
                _, computed_cluster_labels_cosine = faiss_search_index.search(features_cosine, 1)



            """============ Compute Nearest Neighbours ==============="""
            if 'nearest_features' in self.requires:
                faiss_search_index  = faiss.IndexFlatL2(features.shape[-1])
                if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(features)

                max_kval            = np.max([int(x.split('@')[-1]) for x in self.metric_names if ('recall' in x) or ('logl' in x)])
                _, k_closest_points = faiss_search_index.search(features, int(max_kval+1))
                k_closest_classes   = target_labels.reshape(-1)[k_closest_points[:,1:]]

            if 'nearest_features_cosine' in self.requires:
                faiss_search_index  = faiss.IndexFlatIP(features_cosine.shape[-1])
                if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(normalize(features_cosine, axis=1))

                max_kval                   = np.max([int(x.split('@')[-1]) for x in self.metric_names if ('recall' in x) or ('logl' in x)])
                _, k_closest_points_cosine = faiss_search_index.search(normalize(features_cosine, axis=1), int(max_kval+1))
                k_closest_classes_cosine   = target_labels.reshape(-1)[k_closest_points_cosine[:,1:]]


            """============= NED Confidences ================"""
            if ('target_label_prob_ned' in self.requires) or ('label_pred_ned' in self.requires):
                weights = np.zeros((features.shape[0], max_kval))
                for i in np.arange(weights.shape[0]):
                    if 'proxyvmf' in self.pars.loss:
                        # Each neighbour receives exp(log_pcc) as a weight
                        norms = torch.from_numpy(np.linalg.norm(features[k_closest_points_cosine[i,:],:], axis=1)).unsqueeze(1)
                        this_feat = torch.from_numpy(features[i,:]) / norms[0, 0]
                        neighbour_feats = torch.from_numpy(features[k_closest_points_cosine[i,:],:]) / norms
                        # Normalize here, too, if you also normalize the images during training
                        norms = torch.ones(norms.shape) * 20
                        weights[i, ] = np.exp(log_ppk_vmf_vec(mu1=this_feat, kappa1=norms[0, 0], mu2=neighbour_feats, kappa2=norms, rho=self.pars.loss_proxyvmf_rho))[1:]
                    else:
                        # Each neighbour receives exp(-(L2-dist) / temp) as a weight
                        # [1:] because the closest image will always be the image itself
                        weights[i, ] = np.exp(-np.linalg.norm(features[i,:] - features[k_closest_points[i,:],:], axis=1)/self.pars.temp_ned)[1:]

                # calculate probabilities of true label and predicted label
                # TODO: Rework code (closest neighbours, closest neighbours cosine, or even closest points ppk?)
                label_pred_ned = np.zeros((features.shape[0]))
                target_label_prob_ned = np.zeros((features.shape[0]))
                for i in np.arange(weights.shape[0]):
                    neighbour_labels = target_labels[k_closest_points_cosine[i,:], 0] # TODO: cloest_points_cosine?
                    candidate_labels = np.unique(neighbour_labels)
                    candidate_labels_prob = np.zeros(candidate_labels.shape[0])
                    for c in np.arange(candidate_labels.shape[0]):
                        candidate_labels_prob[c] = np.sum(weights[i,neighbour_labels[1:] == candidate_labels[c]]) / \
                                                   np.sum(weights[i,:])
                    label_pred_ned[i] = candidate_labels[np.argmax(candidate_labels_prob)]
                    #label_pred_ned[i] = neighbour_labels[np.argmax(weights[i,:]) + 1]
                    target_label_prob_ned[i] = candidate_labels_prob[candidate_labels == target_labels[i]]


            ###
            if self.pars.evaluate_on_gpu:
                features        = torch.from_numpy(features).to(self.pars.device)
                features_cosine = torch.from_numpy(features_cosine).to(self.pars.device)

            start = time.time()
            for metric in self.list_of_metrics:
                input_dict = {}
                if 'features' in metric.requires:         input_dict['features'] = features
                if 'target_labels' in metric.requires:    input_dict['target_labels'] = target_labels
                if 'dataloader' in metric.requires:       input_dict['dataloader'] = dataloader

                if 'kmeans' in metric.requires:           input_dict['centroids'] = centroids
                if 'kmeans_nearest' in metric.requires:   input_dict['computed_cluster_labels'] = computed_cluster_labels
                if 'nearest_features' in metric.requires: input_dict['k_closest_classes'] = k_closest_classes

                if 'features_cosine' in metric.requires:         input_dict['features_cosine'] = features_cosine

                if 'kmeans_cosine' in metric.requires:           input_dict['centroids_cosine'] = centroids_cosine
                if 'kmeans_nearest_cosine' in metric.requires:   input_dict['computed_cluster_labels_cosine'] = computed_cluster_labels_cosine
                if 'nearest_features_cosine' in metric.requires: input_dict['k_closest_classes_cosine'] = k_closest_classes_cosine
                if 'nearest_points_cosine' in metric.requires:   input_dict['nearest_points_cosine'] = k_closest_points_cosine

                if 'label_pred_ned' in metric.requires:         input_dict['label_pred_ned'] = label_pred_ned
                if 'target_label_prob_ned' in metric.requires:  input_dict['target_label_prob_ned'] = target_label_prob_ned

                computed_metrics[evaltype][metric.name] = metric(**input_dict)

            extra_infos[evaltype] = {'features':features, 'target_labels':target_labels,
                                     'image_paths': dataloader.dataset.image_paths,
                                     'query_image_paths':None, 'gallery_image_paths':None}

        torch.cuda.empty_cache()
        return computed_metrics, extra_infos
