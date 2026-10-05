use feat_math::kernel::cumulative_integration::{
    Config, Edit, ExperimentalProfile, IntegrationProgram, IntegrationSearch, Mode,
};
use feat_math::kernel::ordered_axis::{OrderedInputIdentity, TaskMode};
use feat_math::kernel::wire_v2::VersionedProgram;
#[test]
fn imported_initial_words_read_and_drop_but_do_not_escape_the_search_menu() {
    let child = feat_math::GenomeIr::parse("(pipe X (ridge lambda_frac=0.01))").unwrap();
    let input = OrderedInputIdentity::new("group", "time");
    let search =
        IntegrationSearch::new(1, input.clone(), child.clone(), TaskMode::Forecasting).unwrap();
    let profile = ExperimentalProfile {
        integration: true,
        prior: 0.125,
    };
    for initial in [-0.0, 1.0, 1e12] {
        let stored = VersionedProgram::IntegrationV2(
            IntegrationProgram::new(
                Config::new(2, 60, Mode::Rect, vec![initial]).unwrap(),
                input.clone(),
                child.clone(),
            )
            .unwrap(),
        );
        let bytes = stored.to_bytes().unwrap();
        assert_eq!(
            VersionedProgram::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        assert!(search.edit(&stored, 7, Edit::Change, profile).is_err());
        assert_eq!(
            search
                .edit(
                    &stored,
                    7,
                    Edit::Drop,
                    ExperimentalProfile {
                        integration: false,
                        prior: 0.0
                    }
                )
                .unwrap()
                .to_bytes()
                .unwrap(),
            search.base().to_bytes().unwrap()
        );
    }
    let grown = search.edit(&search.base(), 7, Edit::Grow, profile).unwrap();
    assert!(search.edit(&grown, 7, Edit::Change, profile).is_ok());
}
