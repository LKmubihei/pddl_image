(define (problem picture_43)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_regulator blue_regulator blue_battery - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear blue_battery)
        (clear blue_regulator)
        (clear red_regulator)
        (part_at red_regulator table)
        (part_at blue_regulator table)
        (part_at blue_battery table)
    )
    
    (:goal
        (and
            (part_at red_regulator regulator_placement)
            (part_at blue_battery battery_placement)
        )
    )
)
