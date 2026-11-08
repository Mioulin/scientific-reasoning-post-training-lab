import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("analyse")
    rm = sub.add_parser("evaluate-reward")
    rm.add_argument("--adapter", default="checkpoints/reward/final")
    rm.add_argument("--split", default="test")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--name", required=True)
    ev.add_argument("--adapter")
    ev.add_argument("--split", default="test", choices=["validation", "test", "final"])
    ev.add_argument("--limit", type=int)
    tr = sub.add_parser("train")
    tr.add_argument("kind", choices=["sft", "dpo", "reward"])
    tr.add_argument("--config", required=True)
    tr.add_argument("--resume")
    tr.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare

        print(json.dumps(prepare(), indent=2))
    elif args.command == "evaluate":
        from .runtime import evaluate

        evaluate(args.name, args.adapter, args.split, args.limit)
    elif args.command == "analyse":
        from .analysis import analyse

        analyse()
    elif args.command == "evaluate-reward":
        from .reward_evaluation import evaluate_reward

        evaluate_reward(args.adapter, args.split)
    else:
        from .training import train

        train(args.kind, args.config, args.resume, args.smoke)


if __name__ == "__main__":
    main()
