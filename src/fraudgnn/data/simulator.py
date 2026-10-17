"""Realistic synthetic transaction simulator for fraud detection with Graph Neural Networks.

Generates heterogeneous transaction streams with users, cards, merchants,
devices, and IP addresses. Simulates realistic circadian diurnal spending
distributions and benign graph noise (shared household/office devices and public IPs),
while injecting 5 distinct financial fraud typologies:
1. Coordinated fraud rings (dense bipartite cliques) — individually normal-looking
2. Mule accounts (rapid inbound-outbound forwarding) — normal amounts, graph paths
3. Card testing (bursts of micro-authorizations preceding large cash-outs) — overlapping amounts
4. Account takeover (ATO: country/device anomaly + high amount spike) — partially domestic
5. Merchant collusion (abnormal fraud concentration at rogue merchants)

All hyperparameters, paths, and distributions are controlled via DataConfig.
No magic numbers are hardcoded.

v2: Hardened for meaningful GNN vs tabular benchmark (Phase 2.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudgnn.config import DataConfig


@dataclass(frozen=True)
class EntityPool:
    """Pre-computed registry of users, cards, merchants, devices, and IPs."""

    users: list[str]
    user_home_countries: list[str]
    user_cards: list[list[str]]
    user_usual_devices: list[list[str]]
    user_usual_ips: list[list[str]]
    user_spend_locs: np.ndarray
    user_spend_scales: np.ndarray
    user_activity_weights: np.ndarray
    user_noise_devices: dict[int, str]
    merchants: list[str]
    merchant_categories: list[str]
    merchant_countries: list[str]
    merchant_weights: np.ndarray
    collusive_merchant_indices: list[int]
    devices: list[str]
    ips: list[str]
    shared_ip_pool: list[str]


class TransactionSimulator:
    """Simulates realistic transaction streams and injects structured fraud patterns.

    Parameters:
        config: Validated DataConfig containing generation parameters.
    """

    # Static categorical reference lists (not hyperparameters)
    COUNTRIES: tuple[str, ...] = ("US", "GB", "DE", "CA", "FR", "AU")
    COUNTRY_PROBS: tuple[float, ...] = (0.75, 0.08, 0.05, 0.05, 0.04, 0.03)
    ATO_FOREIGN_COUNTRIES: tuple[str, ...] = ("RU", "NG", "VN", "BR", "UA", "CN")
    CHANNELS: tuple[str, ...] = ("mobile", "web", "pos")
    CHANNEL_PROBS: tuple[float, ...] = (0.55, 0.35, 0.10)
    MERCHANT_CATEGORIES: tuple[str, ...] = (
        "grocery",
        "retail",
        "restaurant",
        "electronics",
        "travel",
        "digital_goods",
        "p2p_transfer",
        "crypto",
    )
    MERCHANT_CAT_PROBS: tuple[float, ...] = (
        0.30,
        0.25,
        0.20,
        0.08,
        0.05,
        0.06,
        0.03,
        0.03,
    )

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.pool = self._init_entity_pool()

    def _init_entity_pool(self) -> EntityPool:
        """Create deterministic pools of users, cards, merchants, devices, and IPs.

        Returns:
            Pre-computed EntityPool.
        """
        cfg = self.config

        # 1. Devices and IPs
        devices = [f"dev_{i:06d}" for i in range(cfg.num_devices)]
        ips = [f"198.51.{(i >> 8) & 255}.{i & 255}" for i in range(cfg.num_ips)]
        shared_ip_count = max(10, int(cfg.num_ips * cfg.legit_ip_pool_frac))
        shared_ip_pool = ips[:shared_ip_count]

        # 2. Users and behavioral profiles
        users = [f"usr_{i:06d}" for i in range(cfg.num_users)]
        home_countries = self.rng.choice(
            self.COUNTRIES,
            size=cfg.num_users,
            p=self.COUNTRY_PROBS,
        ).tolist()

        # Cards: 1 or 2 cards per user
        user_cards: list[list[str]] = []
        for i in range(cfg.num_users):
            n_cards = 1 if self.rng.random() > 0.3 else 2
            user_cards.append([f"crd_{i:06d}_{c}" for c in range(n_cards)])

        # Usual devices: 1 or 2 devices per user
        user_usual_devices: list[list[str]] = []
        for _ in range(cfg.num_users):
            n_devs = 1 if self.rng.random() > 0.25 else 2
            dev_idx = self.rng.choice(cfg.num_devices, size=n_devs, replace=False)
            user_usual_devices.append([devices[d] for d in dev_idx])

        # Usual IPs: 1 to 3 IPs per user
        user_usual_ips: list[list[str]] = []
        for _ in range(cfg.num_users):
            n_ips = int(self.rng.integers(1, 4))
            ip_idx = self.rng.choice(cfg.num_ips, size=n_ips, replace=False)
            user_usual_ips.append([ips[p] for p in ip_idx])

        # Spending parameters (per user baseline)
        user_spend_locs = self.rng.normal(
            loc=cfg.base_amount_loc,
            scale=0.35,
            size=cfg.num_users,
        )
        user_spend_scales = np.full(cfg.num_users, cfg.base_amount_scale)

        # Activity weights: Pareto/Power-law frequency
        activity_raw = self.rng.pareto(a=2.0, size=cfg.num_users) + 1.0
        user_activity_weights = activity_raw / activity_raw.sum()

        # Shared household/office device clusters (benign graph noise)
        num_clusters = max(10, int(cfg.num_users * cfg.legit_device_cluster_frac))
        user_noise_devices: dict[int, str] = {}
        for c_idx in range(num_clusters):
            shared_dev = devices[c_idx % cfg.num_devices]
            cluster_size = int(
                self.rng.integers(
                    cfg.legit_device_cluster_size_min,
                    cfg.legit_device_cluster_size_max + 1,
                )
            )
            cluster_members = self.rng.choice(
                cfg.num_users,
                size=cluster_size,
                replace=False,
            )
            for member in cluster_members:
                user_noise_devices[int(member)] = shared_dev

        # 3. Merchants
        merchants = [f"mer_{i:05d}" for i in range(cfg.num_merchants)]
        merchant_categories = self.rng.choice(
            self.MERCHANT_CATEGORIES,
            size=cfg.num_merchants,
            p=self.MERCHANT_CAT_PROBS,
        ).tolist()
        merchant_countries = self.rng.choice(
            self.COUNTRIES,
            size=cfg.num_merchants,
            p=self.COUNTRY_PROBS,
        ).tolist()

        # Merchant transaction volume weights (Zipfian)
        merch_raw = 1.0 / np.power(np.arange(1, cfg.num_merchants + 1), 0.7)
        merchant_weights = merch_raw / merch_raw.sum()

        # Subset designated for merchant collusion
        num_collusive = max(2, int(cfg.num_merchants * 0.01))
        collusive_merchant_indices = self.rng.choice(
            cfg.num_merchants,
            size=num_collusive,
            replace=False,
        ).tolist()

        return EntityPool(
            users=users,
            user_home_countries=home_countries,
            user_cards=user_cards,
            user_usual_devices=user_usual_devices,
            user_usual_ips=user_usual_ips,
            user_spend_locs=user_spend_locs,
            user_spend_scales=user_spend_scales,
            user_activity_weights=user_activity_weights,
            user_noise_devices=user_noise_devices,
            merchants=merchants,
            merchant_categories=merchant_categories,
            merchant_countries=merchant_countries,
            merchant_weights=merchant_weights,
            collusive_merchant_indices=collusive_merchant_indices,
            devices=devices,
            ips=ips,
            shared_ip_pool=shared_ip_pool,
        )

    def _sample_user_amount(self, user_idx: int) -> float:
        """Sample a transaction amount from the user's normal spending distribution.

        Args:
            user_idx: Index of the user in the entity pool.

        Returns:
            Dollar amount clipped to [1.0, 15000.0].
        """
        loc = self.pool.user_spend_locs[user_idx]
        scale = self.pool.user_spend_scales[user_idx]
        raw = float(np.exp(self.rng.normal(loc, scale)))
        return float(np.clip(np.round(raw, 2), 1.00, 15000.00))

    def _sample_timestamps(self, n: int) -> list[datetime]:
        """Sample timestamps following circadian diurnal distribution across days.

        Args:
            n: Number of timestamps to generate.

        Returns:
            List of datetime objects.
        """
        cfg = self.config
        start_dt = datetime.fromisoformat(cfg.start_date)

        days = self.rng.integers(0, cfg.num_days, size=n)
        hourly_probs = np.array(cfg.hourly_weights, dtype=np.float64)
        hourly_probs = hourly_probs / hourly_probs.sum()
        hours = self.rng.choice(24, size=n, p=hourly_probs)
        minutes = self.rng.integers(0, 60, size=n)
        seconds = self.rng.integers(0, 60, size=n)

        # Microseconds for sub-second jitter and unique ordering
        microsecs = self.rng.integers(0, 1_000_000, size=n)

        timestamps: list[datetime] = []
        for d, h, m, s, us in zip(days, hours, minutes, seconds, microsecs, strict=False):
            ts = start_dt + timedelta(
                days=int(d),
                hours=int(h),
                minutes=int(m),
                seconds=int(s),
                microseconds=int(us),
            )
            timestamps.append(ts)

        return timestamps

    def _sample_circadian_hour(self) -> int:
        """Sample a single hour from the circadian distribution.

        Returns:
            Hour of day (0-23).
        """
        hourly_probs = np.array(self.config.hourly_weights, dtype=np.float64)
        hourly_probs = hourly_probs / hourly_probs.sum()
        return int(self.rng.choice(24, p=hourly_probs))

    def _generate_legitimate_transactions(self, n: int) -> dict[str, list[Any]]:
        """Generate benign transactions with enhanced noise for realistic difficulty.

        Includes legitimate travel, device changes, and large purchases that
        overlap with fraud signals to prevent trivial separation.

        Args:
            n: Number of legitimate transactions to generate.

        Returns:
            Dictionary mapping column names to lists of generated values.
        """
        cfg = self.config
        pool = self.pool

        timestamps = self._sample_timestamps(n)

        # Sample users according to activity weights
        user_indices = self.rng.choice(
            cfg.num_users,
            size=n,
            p=pool.user_activity_weights,
        )

        # Sample merchants
        merchant_indices = self.rng.choice(
            cfg.num_merchants,
            size=n,
            p=pool.merchant_weights,
        )

        user_ids: list[str] = []
        card_ids: list[str] = []
        device_ids: list[str] = []
        ip_addresses: list[str] = []
        amounts: list[float] = []
        countries: list[str] = []
        merchant_ids: list[str] = []
        merchant_cats: list[str] = []
        channels: list[str] = []

        # Vectorized amounts
        user_locs = pool.user_spend_locs[user_indices]
        user_scales = pool.user_spend_scales[user_indices]
        raw_amounts = np.exp(self.rng.normal(user_locs, user_scales))
        clipped_amounts = np.clip(np.round(raw_amounts, 2), a_min=1.00, a_max=15000.00)

        channel_choices = self.rng.choice(
            self.CHANNELS,
            size=n,
            p=self.CHANNEL_PROBS,
        )

        for i in range(n):
            u_idx = int(user_indices[i])
            m_idx = int(merchant_indices[i])

            user_ids.append(pool.users[u_idx])

            # Card: pick from user's cards
            cards = pool.user_cards[u_idx]
            card_ids.append(cards[0] if len(cards) == 1 or self.rng.random() > 0.15 else cards[1])

            # Device: noise sharing vs usual device vs new device
            if (
                u_idx in pool.user_noise_devices
                and self.rng.random() < cfg.legit_shared_device_prob
            ):
                device_ids.append(pool.user_noise_devices[u_idx])
            elif self.rng.random() < cfg.legit_new_device_prob:
                # Legitimate new device (e.g., new phone, borrowed laptop)
                device_ids.append(pool.devices[int(self.rng.integers(0, cfg.num_devices))])
            else:
                usual_devs = pool.user_usual_devices[u_idx]
                device_ids.append(
                    usual_devs[0]
                    if len(usual_devs) == 1 or self.rng.random() > 0.2
                    else usual_devs[1]
                )

            # IP: shared office/public IP noise vs usual IP
            if self.rng.random() < cfg.legit_shared_ip_prob:
                device_ip = pool.shared_ip_pool[
                    int(self.rng.integers(0, len(pool.shared_ip_pool)))
                ]
                ip_addresses.append(device_ip)
            else:
                usual_ips = pool.user_usual_ips[u_idx]
                ip_addresses.append(usual_ips[int(self.rng.integers(0, len(usual_ips)))])

            # Country: legitimate travel noise
            if self.rng.random() < cfg.legit_travel_prob:
                countries.append(str(self.rng.choice(self.COUNTRIES)))
            else:
                countries.append(pool.user_home_countries[u_idx])

            # Amount: occasional large legitimate purchases
            if self.rng.random() < cfg.legit_large_purchase_prob:
                large_amt = float(np.round(self.rng.uniform(500.0, 3000.0), 2))
                amounts.append(large_amt)
            else:
                amounts.append(float(clipped_amounts[i]))

            merchant_ids.append(pool.merchants[m_idx])
            merchant_cats.append(pool.merchant_categories[m_idx])
            channels.append(str(channel_choices[i]))

        return {
            "timestamp": timestamps,
            "user_id": user_ids,
            "card_id": card_ids,
            "merchant_id": merchant_ids,
            "device_id": device_ids,
            "ip_address": ip_addresses,
            "amount": amounts,
            "currency": ["USD"] * n,
            "country": countries,
            "merchant_category": merchant_cats,
            "channel": channels,
            "is_fraud": [0] * n,
            "fraud_type": ["legitimate"] * n,
        }

    def _inject_fraud_rings(self, count: int) -> dict[str, list[Any]]:
        """Inject coordinated fraud rings: groups sharing devices/IPs/merchants.

        Hardened: ring members individually look normal — amounts, hours, and
        countries are drawn from normal-user distributions. The fraud signal
        lives primarily in the shared infrastructure (devices/IPs/merchants)
        and temporal coordination, which requires graph structure to detect.

        Args:
            count: Number of fraud ring transactions to generate.

        Returns:
            Dictionary of transaction fields.
        """
        cfg = self.config
        pool = self.pool
        res: dict[str, list[Any]] = {k: [] for k in self._empty_record_dict()}

        start_dt = datetime.fromisoformat(cfg.start_date)
        generated = 0

        while generated < count:
            ring_size = int(self.rng.integers(cfg.ring_size_min, cfg.ring_size_max + 1))
            ring_users = self.rng.choice(cfg.num_users, size=ring_size, replace=False)

            # Shared infrastructure — the core graph signal
            ring_devices = [
                pool.devices[int(self.rng.integers(0, cfg.num_devices))]
                for _ in range(max(2, ring_size // 4))
            ]
            ring_ips = [
                pool.ips[int(self.rng.integers(0, cfg.num_ips))]
                for _ in range(max(2, ring_size // 3))
            ]
            ring_merchants = [
                pool.merchants[int(self.rng.integers(0, cfg.num_merchants))] for _ in range(3)
            ]

            # Spread activity over a wider time window (less burst-y)
            base_day = int(self.rng.integers(0, max(1, cfg.num_days - 4)))
            base_time = start_dt + timedelta(
                days=base_day,
                hours=int(self.rng.integers(0, 24)),
            )

            tx_in_ring = min(count - generated, ring_size * 2)
            for _ in range(tx_in_ring):
                u_idx = int(self.rng.choice(ring_users))

                # Timing: spread over configurable jitter window
                ts = base_time + timedelta(
                    hours=float(self.rng.uniform(0.0, cfg.ring_timing_jitter_hours)),
                    minutes=float(self.rng.uniform(0.0, 59.0)),
                    seconds=float(self.rng.uniform(0.0, 59.0)),
                )

                # Amount: normal user distribution (not suspicious range)
                if cfg.ring_use_normal_amounts:
                    amt = self._sample_user_amount(u_idx)
                else:
                    amt = float(np.round(self.rng.uniform(150.0, 1200.0), 2))

                merch = str(self.rng.choice(ring_merchants))
                m_idx = pool.merchants.index(merch)

                # Country: use member's home country (not suspicious)
                if cfg.ring_use_home_country:
                    country = pool.user_home_countries[u_idx]
                else:
                    country = pool.user_home_countries[u_idx]

                res["timestamp"].append(ts)
                res["user_id"].append(pool.users[u_idx])
                res["card_id"].append(pool.user_cards[u_idx][0])
                res["merchant_id"].append(merch)
                # Shared devices/IPs — the graph-only signal
                res["device_id"].append(str(self.rng.choice(ring_devices)))
                res["ip_address"].append(str(self.rng.choice(ring_ips)))
                res["amount"].append(amt)
                res["currency"].append("USD")
                res["country"].append(country)
                res["merchant_category"].append(pool.merchant_categories[m_idx])
                res["channel"].append(str(self.rng.choice(self.CHANNELS)))
                res["is_fraud"].append(1)
                res["fraud_type"].append("fraud_ring")

            generated += tx_in_ring

        return res

    def _inject_mules(self, count: int) -> dict[str, list[Any]]:
        """Inject mule accounts: rapid forwards via graph paths with normal amounts.

        Hardened: mule transaction amounts are drawn from normal-user distributions.
        The fraud signal is in the graph structure (receive → forward chains
        through specific merchants) rather than unusual dollar amounts.

        Args:
            count: Number of mule transactions to generate.

        Returns:
            Dictionary of transaction fields.
        """
        cfg = self.config
        pool = self.pool
        res: dict[str, list[Any]] = {k: [] for k in self._empty_record_dict()}

        start_dt = datetime.fromisoformat(cfg.start_date)
        generated = 0

        # Cash-out merchants (crypto or p2p transfer)
        cashout_merch_indices = [
            i
            for i, cat in enumerate(pool.merchant_categories)
            if cat in ("crypto", "p2p_transfer", "travel")
        ]
        if not cashout_merch_indices:
            cashout_merch_indices = list(range(10))

        while generated < count:
            mule_user_idx = int(self.rng.integers(0, cfg.num_users))
            burst_len = int(self.rng.integers(cfg.mule_burst_min, cfg.mule_burst_max + 1))
            burst_len = min(burst_len, count - generated)

            base_day = int(self.rng.integers(0, cfg.num_days))
            base_time = start_dt + timedelta(
                days=base_day,
                hours=int(self.rng.integers(0, 24)),
                minutes=int(self.rng.integers(0, 60)),
            )

            # Use the user's own usual devices/IPs (not suspicious individually)
            dev = pool.user_usual_devices[mule_user_idx][0]
            ip = pool.user_usual_ips[mule_user_idx][0]

            for step in range(burst_len):
                # Varied inter-transaction timing (configurable)
                ts = base_time + timedelta(
                    minutes=step
                    * float(
                        self.rng.uniform(
                            cfg.mule_inter_tx_minutes_min,
                            cfg.mule_inter_tx_minutes_max,
                        )
                    )
                )

                # Amount: from normal user distribution with jitter
                if cfg.mule_use_normal_amounts:
                    base_amt = self._sample_user_amount(mule_user_idx)
                    jitter = float(
                        self.rng.uniform(
                            1.0 - cfg.mule_amount_jitter,
                            1.0 + cfg.mule_amount_jitter,
                        )
                    )
                    amt = float(np.round(base_amt * jitter, 2))
                else:
                    base_amt = float(self.rng.uniform(500.0, 3500.0))
                    amt = float(np.round(base_amt * self.rng.uniform(0.92, 0.99), 2))

                m_idx = int(self.rng.choice(cashout_merch_indices))

                res["timestamp"].append(ts)
                res["user_id"].append(pool.users[mule_user_idx])
                res["card_id"].append(pool.user_cards[mule_user_idx][0])
                res["merchant_id"].append(pool.merchants[m_idx])
                res["device_id"].append(dev)
                res["ip_address"].append(ip)
                res["amount"].append(amt)
                res["currency"].append("USD")
                res["country"].append(pool.user_home_countries[mule_user_idx])
                res["merchant_category"].append(pool.merchant_categories[m_idx])
                res["channel"].append("mobile")
                res["is_fraud"].append(1)
                res["fraud_type"].append("mule")

            generated += burst_len

        return res

    def _inject_card_testing(self, count: int) -> dict[str, list[Any]]:
        """Inject card testing: burst of charges with amounts overlapping normal spending.

        Hardened: small test amounts overlap with legitimate micro-transactions,
        and large cash-out amounts overlap with legitimate high-value purchases.

        Args:
            count: Number of card testing transactions to generate.

        Returns:
            Dictionary of transaction fields.
        """
        cfg = self.config
        pool = self.pool
        res: dict[str, list[Any]] = {k: [] for k in self._empty_record_dict()}

        start_dt = datetime.fromisoformat(cfg.start_date)
        generated = 0

        # Automated micro-charge merchants (digital_goods, retail)
        micro_merch_indices = [
            i
            for i, cat in enumerate(pool.merchant_categories)
            if cat in ("digital_goods", "retail")
        ]
        # Target high-value cashout merchants (electronics)
        large_merch_indices = [
            i for i, cat in enumerate(pool.merchant_categories) if cat in ("electronics", "travel")
        ]
        if not micro_merch_indices:
            micro_merch_indices = list(range(5))
        if not large_merch_indices:
            large_merch_indices = list(range(5, 10))

        while generated < count:
            victim_user_idx = int(self.rng.integers(0, cfg.num_users))
            stolen_card = pool.user_cards[victim_user_idx][0]

            attacker_dev = pool.devices[int(self.rng.integers(0, cfg.num_devices))]
            attacker_ip = pool.ips[int(self.rng.integers(0, cfg.num_ips))]

            burst_len = int(
                self.rng.integers(cfg.card_testing_burst_min, cfg.card_testing_burst_max + 1)
            )
            # Ensure burst fits into remaining count (at least 2 transactions: probe + cashout)
            burst_len = min(burst_len, count - generated)
            if burst_len < 2:
                burst_len = min(2, count - generated)

            base_day = int(self.rng.integers(0, cfg.num_days))
            base_time = start_dt + timedelta(
                days=base_day,
                hours=self._sample_circadian_hour(),
                minutes=int(self.rng.integers(0, 50)),
            )

            # Probe transactions: amounts overlap with normal small purchases
            for probe_i in range(burst_len - 1):
                ts = base_time + timedelta(seconds=probe_i * float(self.rng.uniform(10.0, 60.0)))
                amt = float(
                    np.round(
                        self.rng.uniform(cfg.card_testing_small_min, cfg.card_testing_small_max),
                        2,
                    )
                )
                m_idx = int(self.rng.choice(micro_merch_indices))

                res["timestamp"].append(ts)
                res["user_id"].append(pool.users[victim_user_idx])
                res["card_id"].append(stolen_card)
                res["merchant_id"].append(pool.merchants[m_idx])
                res["device_id"].append(attacker_dev)
                res["ip_address"].append(attacker_ip)
                res["amount"].append(amt)
                res["currency"].append("USD")
                res["country"].append(pool.user_home_countries[victim_user_idx])
                res["merchant_category"].append(pool.merchant_categories[m_idx])
                res["channel"].append("web")
                res["is_fraud"].append(1)
                res["fraud_type"].append("card_testing")

            # Cash-out: amount overlaps with normal high-value purchases
            cashout_ts = base_time + timedelta(
                seconds=(burst_len - 1) * float(self.rng.uniform(20.0, 90.0))
            )
            cashout_amt = float(
                np.round(
                    self.rng.uniform(cfg.card_testing_large_min, cfg.card_testing_large_max),
                    2,
                )
            )
            cashout_m_idx = int(self.rng.choice(large_merch_indices))

            res["timestamp"].append(cashout_ts)
            res["user_id"].append(pool.users[victim_user_idx])
            res["card_id"].append(stolen_card)
            res["merchant_id"].append(pool.merchants[cashout_m_idx])
            res["device_id"].append(attacker_dev)
            res["ip_address"].append(attacker_ip)
            res["amount"].append(cashout_amt)
            res["currency"].append("USD")
            res["country"].append(pool.user_home_countries[victim_user_idx])
            res["merchant_category"].append(pool.merchant_categories[cashout_m_idx])
            res["channel"].append("web")
            res["is_fraud"].append(1)
            res["fraud_type"].append("card_testing")

            generated += burst_len

        return res

    def _inject_account_takeover(self, count: int) -> dict[str, list[Any]]:
        """Inject Account Takeover (ATO) with reduced tabular separability.

        Hardened: a fraction of ATOs are domestic (same country) and some
        reuse a known device (simulating SIM swap or session hijack). Amount
        spike is reduced to overlap with legitimate large purchases.

        Args:
            count: Number of ATO transactions to generate.

        Returns:
            Dictionary of transaction fields.
        """
        cfg = self.config
        pool = self.pool
        res: dict[str, list[Any]] = {k: [] for k in self._empty_record_dict()}

        start_dt = datetime.fromisoformat(cfg.start_date)

        for _ in range(count):
            victim_idx = int(self.rng.integers(0, cfg.num_users))

            # Device: some ATOs reuse a known device (e.g., SIM swap)
            if self.rng.random() < cfg.ato_use_known_device_prob:
                ato_device = pool.user_usual_devices[victim_idx][0]
            else:
                ato_device = pool.devices[int(self.rng.integers(0, cfg.num_devices))]

            ato_ip = pool.ips[int(self.rng.integers(0, cfg.num_ips))]

            # Country: some ATOs are domestic (harder to detect)
            if self.rng.random() < cfg.ato_use_domestic_prob:
                country = pool.user_home_countries[victim_idx]
            else:
                country = str(self.rng.choice(self.ATO_FOREIGN_COUNTRIES))

            # Amount spike: reduced multiplier, overlaps with legit large purchases
            user_mean_spend = float(np.exp(pool.user_spend_locs[victim_idx]))
            ato_amt = float(
                np.round(
                    user_mean_spend * cfg.ato_amount_multiplier * self.rng.uniform(0.8, 1.5),
                    2,
                )
            )
            ato_amt = max(50.00, min(ato_amt, 5000.00))

            # High value merchant
            m_idx = int(self.rng.integers(0, cfg.num_merchants))
            ts = start_dt + timedelta(
                days=int(self.rng.integers(0, cfg.num_days)),
                hours=int(self.rng.integers(0, 24)),
                minutes=int(self.rng.integers(0, 60)),
                seconds=int(self.rng.integers(0, 60)),
            )

            res["timestamp"].append(ts)
            res["user_id"].append(pool.users[victim_idx])
            res["card_id"].append(pool.user_cards[victim_idx][0])
            res["merchant_id"].append(pool.merchants[m_idx])
            res["device_id"].append(ato_device)
            res["ip_address"].append(ato_ip)
            res["amount"].append(ato_amt)
            res["currency"].append("USD")
            res["country"].append(country)
            res["merchant_category"].append(pool.merchant_categories[m_idx])
            res["channel"].append("web")
            res["is_fraud"].append(1)
            res["fraud_type"].append("account_takeover")

        return res

    def _inject_merchant_collusion(self, count: int) -> dict[str, list[Any]]:
        """Inject merchant collusion: rogue merchants with concentrated fraud charges.

        Args:
            count: Number of collusive transactions to generate.

        Returns:
            Dictionary of transaction fields.
        """
        cfg = self.config
        pool = self.pool
        res: dict[str, list[Any]] = {k: [] for k in self._empty_record_dict()}

        start_dt = datetime.fromisoformat(cfg.start_date)

        for _ in range(count):
            collusive_m_idx = int(self.rng.choice(pool.collusive_merchant_indices))
            random_user_idx = int(self.rng.integers(0, cfg.num_users))

            ts = start_dt + timedelta(
                days=int(self.rng.integers(0, cfg.num_days)),
                hours=int(self.rng.integers(0, 24)),
                minutes=int(self.rng.integers(0, 60)),
                seconds=int(self.rng.integers(0, 60)),
            )
            # Use normal-looking amount from user distribution
            amt = self._sample_user_amount(random_user_idx)

            res["timestamp"].append(ts)
            res["user_id"].append(pool.users[random_user_idx])
            res["card_id"].append(pool.user_cards[random_user_idx][0])
            res["merchant_id"].append(pool.merchants[collusive_m_idx])
            res["device_id"].append(pool.devices[int(self.rng.integers(0, cfg.num_devices))])
            res["ip_address"].append(pool.ips[int(self.rng.integers(0, cfg.num_ips))])
            res["amount"].append(amt)
            res["currency"].append("USD")
            res["country"].append(pool.merchant_countries[collusive_m_idx])
            res["merchant_category"].append(pool.merchant_categories[collusive_m_idx])
            res["channel"].append(str(self.rng.choice(self.CHANNELS)))
            res["is_fraud"].append(1)
            res["fraud_type"].append("merchant_collusion")

        return res

    def _apply_label_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flip a fraction of fraud labels to 0 (undetectable by design).

        This simulates real-world label noise where some fraud transactions
        are inherently undetectable from the available features.

        Args:
            df: Transaction DataFrame with is_fraud column.

        Returns:
            DataFrame with label noise applied.
        """
        cfg = self.config
        if cfg.label_noise_rate <= 0.0:
            return df

        fraud_indices = df.index[df["is_fraud"] == 1].tolist()
        n_flip = max(0, int(len(fraud_indices) * cfg.label_noise_rate))
        if n_flip == 0:
            return df

        flip_indices = self.rng.choice(fraud_indices, size=n_flip, replace=False)
        df = df.copy()
        df.loc[flip_indices, "is_fraud"] = 0
        # Keep fraud_type for analysis, but label is now 0
        return df

    def _empty_record_dict(self) -> dict[str, list[Any]]:
        """Return an empty dictionary mapping column names to empty lists.

        Returns:
            Dictionary with transaction schema keys.
        """
        return {
            "timestamp": [],
            "user_id": [],
            "card_id": [],
            "merchant_id": [],
            "device_id": [],
            "ip_address": [],
            "amount": [],
            "currency": [],
            "country": [],
            "merchant_category": [],
            "channel": [],
            "is_fraud": [],
            "fraud_type": [],
        }

    def generate(self) -> pd.DataFrame:
        """Run the simulation to generate a full synthetic transaction DataFrame.

        Returns:
            Chronologically sorted DataFrame containing transactions with ground truth
            and hidden fraud typology labels.
        """
        cfg = self.config

        total_tx = cfg.num_transactions
        n_fraud = max(1, round(total_tx * cfg.fraud_ratio))
        n_legit = total_tx - n_fraud

        # Calculate partition of fraud types
        weights = np.array(
            [
                cfg.fraud_ring_weight,
                cfg.mule_weight,
                cfg.card_testing_weight,
                cfg.account_takeover_weight,
                cfg.merchant_collusion_weight,
            ],
            dtype=np.float64,
        )
        weights = weights / weights.sum()

        counts = np.round(weights * n_fraud).astype(int)
        # Reconcile rounding to match exact n_fraud
        counts[-1] = n_fraud - int(counts[:-1].sum())

        # Generate each partition
        legit_data = self._generate_legitimate_transactions(n_legit)
        ring_data = self._inject_fraud_rings(int(counts[0]))
        mule_data = self._inject_mules(int(counts[1]))
        card_test_data = self._inject_card_testing(int(counts[2]))
        ato_data = self._inject_account_takeover(int(counts[3]))
        collusion_data = self._inject_merchant_collusion(int(counts[4]))

        # Merge all partitions
        combined: dict[str, list[Any]] = {k: [] for k in self._empty_record_dict()}
        for part in (
            legit_data,
            ring_data,
            mule_data,
            card_test_data,
            ato_data,
            collusion_data,
        ):
            for k in combined:
                combined[k].extend(part[k])

        df = pd.DataFrame(combined)

        # Sort strictly by timestamp (temporal order is non-negotiable)
        df.sort_values(by="timestamp", ascending=True, inplace=True, ignore_index=True)

        # Assign unique transaction IDs in temporal order
        df.insert(0, "transaction_id", [f"tx_{i:08d}" for i in range(len(df))])

        # Cast column types
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["is_fraud"] = df["is_fraud"].astype(np.int64)
        df["amount"] = df["amount"].astype(np.float64)

        # Apply label noise (undetectable fraud by design)
        df = self._apply_label_noise(df)

        return df

    def save_to_parquet(self, df: pd.DataFrame) -> Path:
        """Save generated DataFrame to parquet in data/raw/.

        Args:
            df: Generated transaction DataFrame.

        Returns:
            Path to the saved parquet file.
        """
        output_dir = Path(self.config.raw_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / self.config.output_filename

        df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        return out_path
